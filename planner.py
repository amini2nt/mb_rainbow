from math import inf
import torch
from torch import jit
from torch.nn.utils import vector_to_parameters, parameters_to_vector

import ipdb


# Model-predictive control planner with cross-entropy method and learned transition model
class MPCPlanner(jit.ScriptModule):
  __constants__ = ['action_size', 'planning_horizon', 'optimisation_iters', 'candidates', 'top_candidates', 'min_action', 'max_action']

  def __init__(self, action_size, planning_horizon, optimisation_iters, candidates, top_candidates, transition_model, reward_model, min_action=-inf, max_action=inf, initial_sigma=1):
    super().__init__()
    self.transition_model, self.reward_model = transition_model, reward_model
    self.action_size, self.min_action, self.max_action = action_size, min_action, max_action
    self.initial_sigma = initial_sigma
    self.planning_horizon = planning_horizon
    self.optimisation_iters = optimisation_iters
    self.candidates, self.top_candidates = candidates, top_candidates
    print(self.initial_sigma)

  @jit.script_method
  def forward(self, belief, state):
    B, H, Z = belief.size(0), belief.size(1), state.size(1)
    belief, state = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H), state.unsqueeze(dim=1).expand(B, self.candidates, Z).reshape(-1, Z)
    # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
    action_mean, action_std_dev = torch.zeros(self.planning_horizon, B, 1, self.action_size, device=belief.device), torch.ones(self.planning_horizon, B, 1, self.action_size, device=belief.device) * self.initial_sigma

    for _ in range(self.optimisation_iters):
      # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
      actions = (action_mean + action_std_dev * torch.randn(self.planning_horizon, B, self.candidates, self.action_size, device=action_mean.device)).view(self.planning_horizon, B * self.candidates, self.action_size)  # Sample actions (time x (batch x candidates) x actions)
      actions.clamp_(min=self.min_action, max=self.max_action)  # Clip action range
      # Sample next states
      beliefs, states, _, _ = self.transition_model(state, actions, belief)
      # Calculate expected returns (technically sum of rewards over planning horizon)
      returns = self.reward_model(beliefs.view(-1, H), states.view(-1, Z)).view(self.planning_horizon, -1).sum(dim=0)
      # Re-fit belief to the K best action sequences
      _, topk = returns.reshape(B, self.candidates).topk(self.top_candidates, dim=1, largest=True, sorted=False)
      topk += self.candidates * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)  # Fix indices for unrolled actions
      best_actions = actions[:, topk.view(-1)].reshape(self.planning_horizon, B, self.top_candidates, self.action_size)
      # Update belief with new means and standard deviations
      action_mean, action_std_dev = best_actions.mean(dim=2, keepdim=True), best_actions.std(dim=2, unbiased=False, keepdim=True)
    # Return first action mean µ_t
    return action_mean[0].squeeze(dim=1)


# Model-predictive control planner with cross-entropy method and learned transition model
class POPA1Planner(jit.ScriptModule):
  __constants__ = ['action_size', 'planning_horizon', 'optimisation_iters', 'candidates', 'top_candidates', 'min_action', 'max_action']

  def __init__(self, action_size, planning_horizon, optimisation_iters, candidates, top_candidates, transition_model, reward_model, policy_net, min_action=-inf, max_action=inf, initial_sigma=1):
    super().__init__()
    self.transition_model, self.reward_model = transition_model, reward_model
    self.policy_net = policy_net
    self.action_size, self.min_action, self.max_action = action_size, min_action, max_action
    self.initial_sigma = initial_sigma
    self.planning_horizon = planning_horizon
    self.optimisation_iters = optimisation_iters
    self.candidates, self.top_candidates = candidates, top_candidates

  @jit.script_method
  def forward(self, belief, state):
    init_actions = []
    init_batch_actions = []
    ibelief = belief
    istate = state
    for i in range(0, self.planning_horizon):
      actions = self.policy_net(ibelief, istate)
      actions.clamp_(min=self.min_action, max=self.max_action)  # Clip action range
      init_actions.append(actions)
      init_batch_actions.append(actions.unsqueeze(1).expand(actions.shape[0], self.candidates, actions.shape[1]).reshape(-1, actions.shape[1]))
      ibelief, istate, _, _ = self.transition_model(istate, init_actions[-1].unsqueeze(0), ibelief)
      ibelief = ibelief.squeeze(0)
      istate = istate.squeeze(0)

    init_actions = torch.stack(init_actions)
    init_batch_actions = torch.stack(init_batch_actions)

    B, H, Z = belief.size(0), belief.size(1), state.size(1)
    belief, state = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H), state.unsqueeze(dim=1).expand(B, self.candidates, Z).reshape(-1, Z)
    # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
    noise_mean, noise_std_dev = torch.zeros(self.planning_horizon, B, 1, self.action_size, device=belief.device), torch.ones(self.planning_horizon, B, 1, self.action_size, device=belief.device)* self.initial_sigma
    for _ in range(self.optimisation_iters):
      # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
      noises = (noise_mean + noise_std_dev * torch.randn(self.planning_horizon, B, self.candidates, self.action_size, device=noise_mean.device)).view(self.planning_horizon, B * self.candidates, self.action_size)  # Sample actions (time x (batch x candidates) x actions)
      actions = init_batch_actions + noises
      actions.clamp_(min=self.min_action, max=self.max_action)  # Clip action range
      # Sample next states
      beliefs, states, _, _ = self.transition_model(state, actions, belief)
      # Calculate expected returns (technically sum of rewards over planning horizon)
      returns = self.reward_model(beliefs.view(-1, H), states.view(-1, Z)).view(self.planning_horizon, -1).sum(dim=0)
      # Re-fit belief to the K best action sequences
      _, topk = returns.reshape(B, self.candidates).topk(self.top_candidates, dim=1, largest=True, sorted=False)
      topk += self.candidates * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)  # Fix indices for unrolled actions
      best_noises = noises[:, topk.view(-1)].reshape(self.planning_horizon, B, self.top_candidates, self.action_size)
      # Update belief with new means and standard deviations
      noise_mean, noise_std_dev = best_noises.mean(dim=2, keepdim=True), best_noises.std(dim=2, unbiased=False, keepdim=True)
    # Return first action mean µ_t
    actions = init_actions[0] + noise_mean[0].squeeze(dim=1)
    actions.clamp_(min=self.min_action, max=self.max_action)  # Clip action range
    return actions


# Model-predictive control planner with cross-entropy method and learned transition model
class POPA2Planner(jit.ScriptModule):
  __constants__ = ['action_size', 'planning_horizon', 'optimisation_iters', 'candidates', 'top_candidates', 'min_action', 'max_action']

  def __init__(self, action_size, planning_horizon, optimisation_iters, candidates, top_candidates, transition_model, reward_model, policy_net, min_action=-inf, max_action=inf, initial_sigma=1):
    super().__init__()
    self.transition_model, self.reward_model = transition_model, reward_model
    self.policy_net = policy_net
    self.action_size, self.min_action, self.max_action = action_size, min_action, max_action
    self.initial_sigma = initial_sigma
    self.planning_horizon = planning_horizon
    self.optimisation_iters = optimisation_iters
    self.candidates, self.top_candidates = candidates, top_candidates

  @jit.script_method
  def forward(self, belief, state):

    init_actions = self.policy_net(belief, state)
    
    B, H, Z = belief.size(0), belief.size(1), state.size(1)
    belief, state = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H), state.unsqueeze(dim=1).expand(B, self.candidates, Z).reshape(-1, Z)
    init_state = state
    init_belief = belief
    # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
    noise_mean, noise_std_dev = torch.zeros(self.planning_horizon, B, 1, self.action_size, device=belief.device), torch.ones(self.planning_horizon, B, 1, self.action_size, device=belief.device)* self.initial_sigma
    for _ in range(self.optimisation_iters):
      # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
      noises = (noise_mean + noise_std_dev * torch.randn(self.planning_horizon, B, self.candidates, self.action_size, device=noise_mean.device)).view(self.planning_horizon, B * self.candidates, self.action_size)  # Sample actions (time x (batch x candidates) x actions)

      beliefs = [torch.empty(0)]*self.planning_horizon
      states = [torch.empty(0)]*self.planning_horizon
      belief = init_belief
      state = init_state
      for i in range(0, self.planning_horizon):
        actions = self.policy_net(belief, state)
        actions = actions + noises[i]
        actions.clamp_(min=self.min_action, max=self.max_action)  # Clip action range
        belief, state, _, _ = self.transition_model(state, actions.unsqueeze(0), belief)
        belief = belief.squeeze(0)
        state = state.squeeze(0)
        beliefs[i] = belief
        states[i] = state

      beliefs = torch.stack(beliefs)
      states = torch.stack(states)

      # Calculate expected returns (technically sum of rewards over planning horizon)
      returns = self.reward_model(beliefs.view(-1, H), states.view(-1, Z)).view(self.planning_horizon, -1).sum(dim=0)
      # Re-fit belief to the K best action sequences
      _, topk = returns.reshape(B, self.candidates).topk(self.top_candidates, dim=1, largest=True, sorted=False)
      topk += self.candidates * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)  # Fix indices for unrolled actions
      best_noises = noises[:, topk.view(-1)].reshape(self.planning_horizon, B, self.top_candidates, self.action_size)
      # Update belief with new means and standard deviations
      noise_mean, noise_std_dev = best_noises.mean(dim=2, keepdim=True), best_noises.std(dim=2, unbiased=False, keepdim=True)
    # Return first action mean µ_t
    actions = init_actions + noise_mean[0].squeeze(dim=1)
    actions.clamp_(min=self.min_action, max=self.max_action)  # Clip action range
    return actions


"""
# Model-predictive control planner with cross-entropy method and learned transition model
class POP_P_Planner(jit.ScriptModule):
  __constants__ = ['action_size', 'planning_horizon', 'optimisation_iters', 'candidates', 'top_candidates', 'min_action', 'max_action']

  def __init__(self, action_size, planning_horizon, optimisation_iters, candidates, top_candidates, transition_model, reward_model, policy_net, min_action=-inf, max_action=inf, initial_sigma=1):
    super().__init__()
    self.transition_model, self.reward_model = transition_model, reward_model
    self.policy_net = policy_net
    self.action_size, self.min_action, self.max_action = action_size, min_action, max_action
    self.initial_sigma = initial_sigma
    self.planning_horizon = planning_horizon
    self.optimisation_iters = optimisation_iters
    self.candidates, self.top_candidates = candidates, top_candidates

  #@jit.script_method
  def forward(self, belief, state):

    init_actions = self.policy_net(belief, state)
    
    B, H, Z = belief.size(0), belief.size(1), state.size(1)
    belief, state = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H), state.unsqueeze(dim=1).expand(B, self.candidates, Z).reshape(-1, Z)
    init_state = state
    init_belief = belief
    param_vector = parameters_to_vector(self.policy_net.parameters())
    num_params = len(param_vector)

    # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
    noise_mean, noise_std_dev = torch.zeros(self.planning_horizon, B, 1, num_params, device=belief.device), torch.ones(self.planning_horizon, B, 1, num_params, device=belief.device)* self.initial_sigma

    for _ in range(self.optimisation_iters):
      # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
      noises = (noise_mean + noise_std_dev * torch.randn(self.planning_horizon, B, self.candidates, num_params, device=noise_mean.device)).view(self.planning_horizon, B * self.candidates, num_params)  # Sample actions (time x (batch x candidates) x actions)
      #ipdb.set_trace()    

      beliefs = [torch.empty(0)]*self.planning_horizon
      states = [torch.empty(0)]*self.planning_horizon
      belief = init_belief
      state = init_state
      for i in range(0, self.planning_horizon):
        actions = [torch.empty(0)]*len(belief)
        for j in range(len(belief)):
          param_vector = parameters_to_vector(self.policy_net.parameters())
          param_vector.add_(noises[i][j])
          vector_to_parameters(param_vector, self.policy_net.parameters())
          action = self.policy_net(belief[j].unsqueeze(0), state[j].unsqueeze(0))
          actions[j] = action
          param_vector.add_(-noises[i][j])
          vector_to_parameters(param_vector, self.policy_net.parameters())
        print(i)
        actions = torch.stack(actions)
        actions = actions.squeeze(1)
        actions.clamp_(min=self.min_action, max=self.max_action)  # Clip action range
        belief, state, _, _ = self.transition_model(state, actions.unsqueeze(0), belief)
        belief = belief.squeeze(0)
        state = state.squeeze(0)
        beliefs[i] = belief
        states[i] = state

      beliefs = torch.stack(beliefs)
      states = torch.stack(states)

      # Calculate expected returns (technically sum of rewards over planning horizon)
      returns = self.reward_model(beliefs.view(-1, H), states.view(-1, Z)).view(self.planning_horizon, -1).sum(dim=0)
      # Re-fit belief to the K best action sequences
      _, topk = returns.reshape(B, self.candidates).topk(self.top_candidates, dim=1, largest=True, sorted=False)
      topk += self.candidates * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)  # Fix indices for unrolled actions
      best_noises = noises[:, topk.view(-1)].reshape(self.planning_horizon, B, self.top_candidates, num_params)
      # Update belief with new means and standard deviations
      noise_mean, noise_std_dev = best_noises.mean(dim=2, keepdim=True), best_noises.std(dim=2, unbiased=False, keepdim=True)
      del noises
      torch.cuda.empty_cache()
    # Return first action mean µ_t
    actions = self.policy_net(init_belief, init_state)
    actions.clamp_(min=self.min_action, max=self.max_action)  # Clip action range
    return actions
"""


# Model-predictive control planner with cross-entropy method and learned transition model
class POP_P_Planner(jit.ScriptModule):
  __constants__ = ['action_size', 'planning_horizon', 'optimisation_iters', 'candidates', 'top_candidates', 'min_action', 'max_action']

  def __init__(self, action_size, planning_horizon, optimisation_iters, candidates, top_candidates, transition_model, reward_model, policy_net, min_action=-inf, max_action=inf, initial_sigma=1):
    super().__init__()
    self.transition_model, self.reward_model = transition_model, reward_model
    self.policy_net = policy_net
    self.action_size, self.min_action, self.max_action = action_size, min_action, max_action
    self.initial_sigma = initial_sigma
    self.planning_horizon = planning_horizon
    self.optimisation_iters = optimisation_iters
    self.candidates, self.top_candidates = candidates, top_candidates

  @jit.script_method
  def forward(self, belief, state):
    
    istate = state
    ibelief = belief
    B, H, Z = belief.size(0), belief.size(1), state.size(1)
    belief, state = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H), state.unsqueeze(dim=1).expand(B, self.candidates, Z).reshape(-1, Z)
    init_state = state
    init_belief = belief
    
    num_params = self.policy_net.num_units
    # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
    noise_mean, noise_std_dev = torch.zeros(self.planning_horizon, B, 1, num_params, device=belief.device), torch.ones(self.planning_horizon, B, 1, num_params, device=belief.device)* self.initial_sigma
    for _ in range(self.optimisation_iters):
      # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
      noises = (noise_mean + noise_std_dev * torch.randn(self.planning_horizon, B, self.candidates, num_params, device=noise_mean.device)).view(self.planning_horizon, B * self.candidates, num_params)  # Sample actions (time x (batch x candidates) x actions)

      beliefs = [torch.empty(0)]*self.planning_horizon
      states = [torch.empty(0)]*self.planning_horizon
      belief = init_belief
      state = init_state
      for i in range(0, self.planning_horizon):
        actions = self.policy_net(belief, state, noises[i])
        actions.clamp_(min=self.min_action, max=self.max_action)  # Clip action range
        belief, state, _, _ = self.transition_model(state, actions.unsqueeze(0), belief)
        belief = belief.squeeze(0)
        state = state.squeeze(0)
        beliefs[i] = belief
        states[i] = state

      beliefs = torch.stack(beliefs)
      states = torch.stack(states)

      # Calculate expected returns (technically sum of rewards over planning horizon)
      returns = self.reward_model(beliefs.view(-1, H), states.view(-1, Z)).view(self.planning_horizon, -1).sum(dim=0)
      # Re-fit belief to the K best action sequences
      _, topk = returns.reshape(B, self.candidates).topk(self.top_candidates, dim=1, largest=True, sorted=False)
      topk += self.candidates * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)  # Fix indices for unrolled actions
      best_noises = noises[:, topk.view(-1)].reshape(self.planning_horizon, B, self.top_candidates, num_params)
      # Update belief with new means and standard deviations
      noise_mean, noise_std_dev = best_noises.mean(dim=2, keepdim=True), best_noises.std(dim=2, unbiased=False, keepdim=True)
    # Return first action mean µ_t

    actions = self.policy_net(ibelief, istate, noise_mean[0].squeeze(dim=1))
    actions.clamp_(min=self.min_action, max=self.max_action)  # Clip action range
    return actions



# Model-predictive control planner with cross-entropy method and learned transition model
class POP_MP_Planner(jit.ScriptModule):
  __constants__ = ['action_size', 'planning_horizon', 'optimisation_iters', 'candidates', 'top_candidates', 'min_action', 'max_action']

  def __init__(self, action_size, planning_horizon, optimisation_iters, candidates, top_candidates, transition_model, reward_model, policy_net, min_action=-inf, max_action=inf, initial_sigma=1):
    super().__init__()
    self.transition_model, self.reward_model = transition_model, reward_model
    self.policy_net = policy_net
    self.action_size, self.min_action, self.max_action = action_size, min_action, max_action
    self.initial_sigma = initial_sigma
    self.planning_horizon = planning_horizon
    self.optimisation_iters = optimisation_iters
    self.candidates, self.top_candidates = candidates, top_candidates

  #@jit.script_method
  def forward(self, belief, state):
    
    istate = state
    ibelief = belief
    B, H, Z = belief.size(0), belief.size(1), state.size(1)
    belief, state = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H), state.unsqueeze(dim=1).expand(B, self.candidates, Z).reshape(-1, Z)
    init_state = state
    init_belief = belief
    
    num_params = self.policy_net.num_units
    # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
    noise_mean, noise_std_dev = torch.zeros(self.planning_horizon, B, 1, num_params, device=belief.device), torch.ones(self.planning_horizon, B, 1, num_params, device=belief.device)* self.initial_sigma
    for _ in range(self.optimisation_iters):
      # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
      noises = (noise_mean + noise_std_dev * torch.randn(self.planning_horizon, B, self.candidates, num_params, device=noise_mean.device)).view(self.planning_horizon, B * self.candidates, num_params)  # Sample actions (time x (batch x candidates) x actions)

      beliefs = [torch.empty(0)]*self.planning_horizon
      states = [torch.empty(0)]*self.planning_horizon
      belief = init_belief
      state = init_state
      for i in range(0, self.planning_horizon):
        actions = self.policy_net(belief, state, noises[i])
        actions.clamp_(min=self.min_action, max=self.max_action)  # Clip action range
        belief, state, _, _ = self.transition_model(state, actions.unsqueeze(0), belief)
        belief = belief.squeeze(0)
        state = state.squeeze(0)
        beliefs[i] = belief
        states[i] = state

      beliefs = torch.stack(beliefs)
      states = torch.stack(states)

      # Calculate expected returns (technically sum of rewards over planning horizon)
      returns = self.reward_model(beliefs.view(-1, H), states.view(-1, Z)).view(self.planning_horizon, -1).sum(dim=0)
      # Re-fit belief to the K best action sequences
      ipdb.set_trace()
      _, topk = returns.reshape(B, self.candidates).topk(self.top_candidates, dim=1, largest=True, sorted=False)
      topk += self.candidates * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)  # Fix indices for unrolled actions
      best_noises = noises[:, topk.view(-1)].reshape(self.planning_horizon, B, self.top_candidates, num_params)
      # Update belief with new means and standard deviations
      noise_mean, noise_std_dev = best_noises.mean(dim=2, keepdim=True), best_noises.std(dim=2, unbiased=False, keepdim=True)
    # Return first action mean µ_t

    actions = self.policy_net(ibelief, istate, noise_mean[0].squeeze(dim=1))
    actions.clamp_(min=self.min_action, max=self.max_action)  # Clip action range
    return actions
