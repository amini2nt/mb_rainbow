from math import inf
import torch
from torch import jit
import ipdb


# Model-predictive control planner with cross-entropy method and learned transition model
class MPCPlanner(jit.ScriptModule):
  __constants__ = ['action_size', 'planning_horizon', 'optimisation_iters', 'candidates', 'top_candidates', 'min_action', 'max_action']

  def __init__(self, action_size, planning_horizon, optimisation_iters, candidates, top_candidates, transition_model, reward_model, min_action=-inf, max_action=inf):
    super().__init__()
    self.transition_model, self.reward_model = transition_model, reward_model
    self.action_size, self.min_action, self.max_action = action_size, min_action, max_action
    self.planning_horizon = planning_horizon
    self.optimisation_iters = optimisation_iters
    self.candidates, self.top_candidates = candidates, top_candidates

  @jit.script_method
  def forward(self, belief, state):
    #ipdb.set_trace()
    B, H, Z = belief.size(0), belief.size(1), state.size(1)
    belief, state = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H), state.unsqueeze(dim=1).expand(B, self.candidates, Z).reshape(-1, Z)
    # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
    action_mean, action_std_dev = torch.zeros(self.planning_horizon, B, 1, self.action_size, device=belief.device), torch.ones(self.planning_horizon, B, 1, self.action_size, device=belief.device)

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
class POPAPlanner(jit.ScriptModule):
  __constants__ = ['action_size', 'planning_horizon', 'optimisation_iters', 'candidates', 'top_candidates', 'min_action', 'max_action']

  def __init__(self, action_size, planning_horizon, optimisation_iters, candidates, top_candidates, transition_model, reward_model, policy_net, min_action=-inf, max_action=inf):
    super().__init__()
    self.transition_model, self.reward_model = transition_model, reward_model
    self.policy_net = policy_net
    self.action_size, self.min_action, self.max_action = action_size, min_action, max_action
    self.planning_horizon = planning_horizon
    self.optimisation_iters = optimisation_iters
    self.candidates, self.top_candidates = candidates, top_candidates

  @jit.script_method
  def forward(self, belief, state):
    init_actions = []
    init_gactions = []
    ibelief = belief
    istate = state
    for i in range(0, self.planning_horizon):
      acs = self.policy_net(ibelief, istate)
      acs.clamp_(min=self.min_action, max=self.max_action)  # Clip action range
      init_actions.append(acs)
      init_gactions.append(acs.unsqueeze(1).expand(acs.shape[0], self.candidates, acs.shape[1]).reshape(-1, acs.shape[1]))
      ibelief, istate, _, _ = self.transition_model(istate, init_actions[-1].unsqueeze(0), ibelief)
      ibelief = ibelief.squeeze(0)
      istate = istate.squeeze(0)


    init_actions = torch.stack(init_actions)
    init_gactions = torch.stack(init_gactions)

    B, H, Z = belief.size(0), belief.size(1), state.size(1)
    belief, state = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H), state.unsqueeze(dim=1).expand(B, self.candidates, Z).reshape(-1, Z)
    # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
    noise_mean, noise_std_dev = torch.zeros(self.planning_horizon, B, 1, self.action_size, device=belief.device), torch.ones(self.planning_horizon, B, 1, self.action_size, device=belief.device)
    for _ in range(self.optimisation_iters):
      # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
      noises = (noise_mean + noise_std_dev * torch.randn(self.planning_horizon, B, self.candidates, self.action_size, device=noise_mean.device)).view(self.planning_horizon, B * self.candidates, self.action_size)  # Sample actions (time x (batch x candidates) x actions)
      actions = init_gactions + noises #todo: clip noise
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
    return init_actions[0] + noise_mean[0].squeeze(dim=1) #not clamped
