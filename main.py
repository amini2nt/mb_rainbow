import argparse
from math import inf
import os
import numpy as np
import torch
import gym
import mujoco_py
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from env import CONTROL_SUITE_ENVS, Env, GYM_ENVS, EnvBatcher
from memory import ExperienceReplayv2 as ExperienceReplay
from models import bottle, Encoder, ObservationModel, RewardModel, ReturnModel, TransitionModel, PolicyNet, StochPolicyNet, AdvantageModel
from planner import MPCPlanner, POPA1Planner, POPA2Planner, POP_P_Planner, MPPI_Planner, POP_P_MPPI
from utils import lineplot, write_video
import ipdb


# Hyperparameters
parser = argparse.ArgumentParser(description='PlaNet')
parser.add_argument('--id', type=str, default='fcem', help='Experiment ID')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--env', type=str, default='cheetah-run', choices=GYM_ENVS + CONTROL_SUITE_ENVS, help='Gym/Control Suite environment')
parser.add_argument('--symbolic-env', action='store_true', help='Symbolic features')
parser.add_argument('--max-episode-length', type=int, default=1000, metavar='T', help='Max episode length')
parser.add_argument('--experience-size', type=int, default=1000000, metavar='D', help='Experience replay size')  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
parser.add_argument('--activation-function', type=str, default='relu', choices=dir(F), help='Model activation function')
parser.add_argument('--embedding-size', type=int, default=1024, metavar='E', help='Observation embedding size')  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
parser.add_argument('--hidden-size', type=int, default=200, metavar='H', help='Hidden size')
parser.add_argument('--belief-size', type=int, default=200, metavar='H', help='Belief/hidden size')
parser.add_argument('--state-size', type=int, default=30, metavar='Z', help='State/latent size')
parser.add_argument('--action-repeat', type=int, default=4, metavar='R', help='Action repeat')
parser.add_argument('--action-noise', type=float, default=0.3, metavar='ε', help='Action noise')
parser.add_argument('--episodes', type=int, default=1000, metavar='E', help='Total number of episodes')
parser.add_argument('--seed-episodes', type=int, default=5, metavar='S', help='Seed episodes')
parser.add_argument('--collect-interval', type=int, default=100, metavar='C', help='Collect interval')
parser.add_argument('--batch-size', type=int, default=50, metavar='B', help='Batch size')
parser.add_argument('--chunk-size', type=int, default=50, metavar='L', help='Chunk size')
parser.add_argument('--overshooting-distance', type=int, default=50, metavar='D', help='Latent overshooting distance/latent overshooting weight for t = 1')
parser.add_argument('--overshooting-kl-beta', type=float, default=0, metavar='β>1', help='Latent overshooting KL weight for t > 1 (0 to disable)')
parser.add_argument('--overshooting-reward-scale', type=float, default=0, metavar='R>1', help='Latent overshooting reward prediction weight for t > 1 (0 to disable)')
parser.add_argument('--global-kl-beta', type=float, default=0, metavar='βg', help='Global KL weight (0 to disable)')
parser.add_argument('--free-nats', type=float, default=3, metavar='F', help='Free nats')
parser.add_argument('--bit-depth', type=int, default=5, metavar='B', help='Image bit depth (quantisation)')
parser.add_argument('--learning-rate', type=float, default=6e-4, metavar='α', help='Learning rate') 
parser.add_argument('--learning-rate-schedule', type=int, default=0, metavar='αS', help='Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)') 
parser.add_argument('--adam-epsilon', type=float, default=1e-3, metavar='ε', help='Adam optimiser epsilon value') 
# Note that original has a linear learning rate decay, but it seems unlikely that this makes a significant difference
parser.add_argument('--grad-clip-norm', type=float, default=100, metavar='C', help='Gradient clipping norm')
parser.add_argument('--planning-horizon', type=int, default=12, metavar='H', help='Planning horizon distance')
parser.add_argument('--optimisation-iters', type=int, default=10, metavar='I', help='Planning optimisation iterations')
parser.add_argument('--candidates', type=int, default=1000, metavar='J', help='Candidate samples per iteration')
parser.add_argument('--top-candidates', type=int, default=100, metavar='K', help='Number of top candidates to fit')
parser.add_argument('--test', action='store_true', help='Test only')
parser.add_argument('--test-interval', type=int, default=10, metavar='I', help='Test interval (episodes)')
parser.add_argument('--test-episodes', type=int, default=10, metavar='E', help='Number of test episodes')
parser.add_argument('--checkpoint-interval', type=int, default=5000, metavar='I', help='Checkpoint interval (episodes)')
parser.add_argument('--checkpoint-experience', action='store_true', help='Checkpoint experience replay')
parser.add_argument('--models', type=str, default='', metavar='M', help='Load model checkpoint')
parser.add_argument('--experience-replay', type=str, default='', metavar='ER', help='Load experience replay')
parser.add_argument('--render', action='store_true', help='Render environment')
# New set of hyper-parameters
parser.add_argument('--initial-sigma', type=float, default=1, help='Initial sigma value for CEM')
parser.add_argument('--use-policy', type=bool, default=False, help='Use a policy network')
parser.add_argument('--stoch-policy', type=bool, default=False, help='Use a stochastic policy')
parser.add_argument('--detach-policy', type=bool, default=False, help='no updates from policy to model')
parser.add_argument('--use-value', type=bool, default=False, help='Use a value network')
parser.add_argument('--planner', type=str, default='CEM', help='Type of planner')
parser.add_argument('--policy-reduce', type=str, default='mean', help='policy loss reduction')
parser.add_argument('--mppi-gamma', type=float, default=2, help='MPPI gamma')
parser.add_argument('--mppi-beta', type=float, default=0.6, help='MPPI beta')
parser.add_argument('--marwil-kappa', type=float, default=1, help='kappa for marwil')
parser.add_argument('--res-dir', type=str, default='results', help='directory location')



args = parser.parse_args()
args.overshooting_distance = min(args.chunk_size, args.overshooting_distance)  # Overshooting distance cannot be greater than chunk size
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
	print(' ' * 26 + k + ': ' + str(v))


# Setup
results_dir = os.path.join(args.res_dir, args.id)
os.makedirs(results_dir, exist_ok=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and not args.disable_cuda:
	args.device = torch.device('cuda')
	torch.cuda.manual_seed(args.seed)
	print("Using GPU...")
else:
	args.device = torch.device('cpu')
metrics = {'steps': [], 'episodes': [], 'train_rewards': [], 'test_episodes': [], 'test_rewards': [], 
			'observation_loss': [], 'reward_loss': [], 'kl_loss': []}
if args.use_policy:
	metrics['policy_loss'] = []
if args.use_value:
	metrics['value_loss'] = []
	#metrics['advantage_loss'] = []

torch.save(args, os.path.join(results_dir, 'args.pth'))


# Initialise training environment and experience replay memory
env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth)
if args.experience_replay is not '' and os.path.exists(args.experience_replay):
	D = torch.load(args.experience_replay)
	metrics['steps'], metrics['episodes'] = [D.steps] * D.episodes, list(range(1, D.episodes + 1))
elif not args.test:
	D = ExperienceReplay(args.experience_size, args.symbolic_env, env.observation_size, env.action_size, args.bit_depth, args.device)
	# Initialise dataset D with S random seed episodes
	for s in range(1, args.seed_episodes + 1):
		observation, done, t = env.reset(), False, 0
		while not done:
			action = env.sample_random_action()
			next_observation, reward, done = env.step(action)
			D.append(observation, action, reward, done)
			observation = next_observation
			t += 1
		metrics['steps'].append(t * args.action_repeat + (0 if len(metrics['steps']) == 0 else metrics['steps'][-1]))
		metrics['episodes'].append(s)


# Initialise model parameters randomly
transition_model = TransitionModel(args.belief_size, args.state_size, env.action_size, args.hidden_size, args.embedding_size, args.activation_function).to(device=args.device)
observation_model = ObservationModel(args.symbolic_env, env.observation_size, args.belief_size, args.state_size, args.embedding_size, args.activation_function).to(device=args.device)
reward_model = RewardModel(args.belief_size, args.state_size, args.hidden_size, args.activation_function).to(device=args.device)
encoder = Encoder(args.symbolic_env, env.observation_size, args.embedding_size, args.activation_function).to(device=args.device)
param_list = list(transition_model.parameters()) + list(observation_model.parameters()) + list(reward_model.parameters()) + list(encoder.parameters())

if args.use_policy:
	if args.stoch_policy:
		policy_net = StochPolicyNet(args.belief_size, args.state_size, args.hidden_size, env.action_size, args.activation_function).to(device=args.device)
	else:
		policy_net = PolicyNet(args.belief_size, args.state_size, args.hidden_size, env.action_size, args.activation_function).to(device=args.device)
	if args.detach_policy:
		policy_optimizer = optim.Adam(policy_net.parameters(), lr=1e-3, eps=args.adam_epsilon)	
	else:
		param_list += list(policy_net.parameters())

if args.use_value:
	value_net = ReturnModel(args.belief_size, args.state_size, args.hidden_size, args.activation_function).to(device=args.device)
	#adv_net = AdvantageModel(args.belief_size, args.state_size, args.hidden_size, env.action_size, args.activation_function).to(device=args.device)
	if args.detach_policy:
		params = list(value_net.parameters()) #+ list(adv_net.parameters())
		value_optimizer = optim.Adam(params, lr=2e-3, eps=args.adam_epsilon)
	else:
		param_list += list(value_net.parameters())
		#param_list += list(adv_net.parameters())

optimiser = optim.Adam(param_list, lr=0 if args.learning_rate_schedule != 0 else args.learning_rate, eps=args.adam_epsilon)


if args.use_value:
	policy_net.ma_adv_norm = torch.tensor([0], dtype=torch.float32, requires_grad=False, device=args.device)


if args.models is not '' and os.path.exists(args.models):
	model_dicts = torch.load(args.models)
	transition_model.load_state_dict(model_dicts['transition_model'])
	observation_model.load_state_dict(model_dicts['observation_model'])
	reward_model.load_state_dict(model_dicts['reward_model'])
	encoder.load_state_dict(model_dicts['encoder'])
	optimiser.load_state_dict(model_dicts['optimiser'])
	if args.use_policy:
		policy_net.load_state_dict(model_dicts['policy_net'])
		if args.detach_policy:
			policy_optimizer.load_state_dict(model_dicts['policy_optimizer'])
	if args.use_value:
		value_net.load_state_dict(model_dicts['value_net'])
		#adv_net.load_state_dict(model_dicts['adv_net'])
		if args.detach_policy:
			value_optimizer.load_state_dict(model_dicts['value_optimizer'])

if args.planner == "CEM":
	planner = MPCPlanner(env.action_size, args.planning_horizon, args.optimisation_iters, args.candidates, args.top_candidates, transition_model, reward_model, env.action_range[0], env.action_range[1], args.initial_sigma)
elif args.planner == "POPA1Planner":
	assert(args.use_policy == True)
	planner = POPA1Planner(env.action_size, args.planning_horizon, args.optimisation_iters, args.candidates, args.top_candidates, transition_model, reward_model, policy_net, env.action_range[0], env.action_range[1], args.initial_sigma)
elif args.planner == "POPA2Planner":
	assert(args.use_policy == True)
	planner = POPA2Planner(env.action_size, args.planning_horizon, args.optimisation_iters, args.candidates, args.top_candidates, transition_model, reward_model, policy_net, env.action_range[0], env.action_range[1], args.initial_sigma)
elif args.planner == "POP_P_Planner":
	assert(args.use_policy == True)
	planner = POP_P_Planner(env.action_size, args.planning_horizon, args.optimisation_iters, args.candidates, args.top_candidates, transition_model, reward_model, policy_net, env.action_range[0], env.action_range[1], args.initial_sigma)
elif args.planner == "POP_MP_Planner":
	assert(args.use_policy == True)
	planner = POP_MP_Planner(env.action_size, args.planning_horizon, args.optimisation_iters, args.candidates, args.top_candidates, transition_model, reward_model, policy_net, env.action_range[0], env.action_range[1], args.initial_sigma)
elif args.planner == "MPPI_Planner":
	planner = MPPI_Planner(env.action_size, args.planning_horizon, args.optimisation_iters, args.candidates, args.top_candidates, transition_model, reward_model, env.action_range[0], env.action_range[1], args.initial_sigma, args.mppi_gamma, args.mppi_beta)
elif args.planner == "POP_P_MPPI":
	planner = POP_P_MPPI(env.action_size, args.planning_horizon, args.optimisation_iters, args.candidates, args.top_candidates, transition_model, reward_model, policy_net, env.action_range[0], env.action_range[1], args.initial_sigma, args.mppi_gamma, args.mppi_beta)


global_prior = Normal(torch.zeros(args.batch_size, args.state_size, device=args.device), torch.ones(args.batch_size, args.state_size, device=args.device))  # Global prior N(0, I)
free_nats = torch.full((1, ), args.free_nats, dtype=torch.float32, device=args.device)  # Allowed deviation in KL divergence


def update_belief_and_act(args, env, planner, transition_model, encoder, belief, posterior_state, action, observation, min_action=-inf, max_action=inf, explore=False):
	# Infer belief over current state q(s_t|o≤t,a<t) from the history
	belief, _, _, _, posterior_state, _, _ = transition_model(posterior_state, action.unsqueeze(dim=0), belief, encoder(observation).unsqueeze(dim=0))  # Action and observation need extra time dimension
	belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)  # Remove time dimension from belief/state
	action = planner(belief, posterior_state)  # Get action from planner(q(s_t|o≤t,a<t), p)
	if explore:
		action = action + np.sqrt(args.action_noise) * torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action
	action.clamp_(min=min_action, max=max_action)  # Clip action range
	next_observation, reward, done = env.step(action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu())  # Perform environment step (action repeats handled internally)
	return belief, posterior_state, action, next_observation, reward, done


# Testing only
if args.test:
	# Set models to eval mode
	transition_model.eval()
	reward_model.eval()
	encoder.eval()
	with torch.no_grad():
		total_reward = 0
		for _ in tqdm(range(args.test_episodes)):
			observation = env.reset()
			belief, posterior_state, action = torch.zeros(1, args.belief_size, device=args.device), torch.zeros(1, args.state_size, device=args.device), torch.zeros(1, env.action_size, device=args.device)
			pbar = tqdm(range(args.max_episode_length // args.action_repeat))
			for t in pbar:
				belief, posterior_state, action, observation, reward, done = update_belief_and_act(args, env, planner, transition_model, encoder, belief, posterior_state, action, observation.to(device=args.device), env.action_range[0], env.action_range[1])
				total_reward += reward
				if args.render:
					env.render()
				if done:
					pbar.close()
					break
	print('Average Reward:', total_reward / args.test_episodes)
	env.close()
	quit()


# Training (and testing)
for episode in tqdm(range(metrics['episodes'][-1] + 1, args.episodes + 1), total=args.episodes, initial=metrics['episodes'][-1] + 1):
	# Model fitting
	
	losses = []
	for s in tqdm(range(args.collect_interval)):
		# Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
		observations, actions, rewards, nonterminals, returns = D.sample(args.batch_size, args.chunk_size)  # Transitions start at time t = 0
		# Create initial belief and state for time t = 0
		init_belief, init_state = torch.zeros(args.batch_size, args.belief_size, device=args.device), torch.zeros(args.batch_size, args.state_size, device=args.device)
		encoded_obs = bottle(encoder, (observations, ))
		# Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
		beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = transition_model(init_state, actions[:-1], init_belief, encoded_obs[1:], nonterminals[:-1])
		# Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
		observation_loss = F.mse_loss(bottle(observation_model, (beliefs, posterior_states)), observations[1:], reduction='none').sum(dim=2 if args.symbolic_env else (2, 3, 4)).mean(dim=(0, 1))
		reward_loss = F.mse_loss(bottle(reward_model, (beliefs, posterior_states)), rewards[:-1], reduction='none').mean(dim=(0, 1))
		kl_loss = torch.max(kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2), free_nats).mean(dim=(0, 1))  # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out


		
		# Apply linearly ramping learning rate schedule
		if args.learning_rate_schedule != 0:
			for group in optimiser.param_groups:
				group['lr'] = min(group['lr'] + args.learning_rate / args.learning_rate_schedule, args.learning_rate)
		# Update model parameters
		
		total_loss = observation_loss + reward_loss + kl_loss
		"""if args.use_policy and not args.detach_policy and not args.stoch_policy:
			
			policy_loss = F.mse_loss(bottle(policy_net, (beliefs, posterior_states)), actions[1:], reduction='none')
			if args.policy_reduce == 'sum':
				policy_loss = policy_loss.sum()
			else:
				policy_loss = policy_loss.mean()
			total_loss += policy_loss"""

		if args.use_policy:

			if args.detach_policy:
				beliefs = beliefs.detach()
				posterior_states = posterior_states.detach()

			if args.stoch_policy:
				policy_loss = -bottle(policy_net.logp, (beliefs, posterior_states, actions[1:]))#.sum(2)
			else:
				policy_loss = F.mse_loss(bottle(policy_net, (beliefs, posterior_states)), actions[1:], reduction='none')

			if args.use_value:
				pred_return = bottle(value_net, (beliefs, posterior_states))
				return_loss = F.mse_loss(pred_return, returns[1:], reduction='none').mean(dim=(0, 1))
				adv_target = returns[1:] - pred_return.detach()
				#pred_adv = bottle(adv_net, (beliefs, posterior_states, actions[1:]))
				#adv_loss = F.mse_loss(pred_adv, adv_target, reduction='none').mean(dim=(0, 1))
				
				if args.detach_policy:
					value_optimizer.zero_grad()
					value_loss = return_loss #+ adv_loss
					value_loss.backward()
					nn.utils.clip_grad_norm_(value_net.parameters(), args.grad_clip_norm, norm_type=2)
					#nn.utils.clip_grad_norm_(adv_net.parameters(), args.grad_clip_norm, norm_type=2)
					value_optimizer.step()

				else:

					total_loss += return_loss #+ adv_loss

				# Update averaged advantage norm.
				policy_net.ma_adv_norm.add_(1e-8 * (torch.mean(torch.pow(adv_target, 2.0)) - policy_net.ma_adv_norm))
				#policy_net.ma_adv_norm = 0.9 * policy_net.ma_adv_norm + 0.1 * torch.mean(torch.pow(adv_target, 2.0))
				# #xponentially weighted advantages.
				exp_advs = torch.exp(args.marwil_kappa *(adv_target / (1e-8 + torch.pow(policy_net.ma_adv_norm, 0.5)))).clamp_(max=20)

				#if not args.stoch_policy:
				exp_advs = exp_advs.unsqueeze(2)

				policy_loss = policy_loss * exp_advs.detach()
			if args.policy_reduce == 'sum':
				policy_loss = policy_loss.sum()
			else:
				policy_loss = policy_loss.mean()
			
			if args.detach_policy:
				policy_optimizer.zero_grad()
				policy_loss.backward()
				nn.utils.clip_grad_norm_(policy_net.parameters(), args.grad_clip_norm, norm_type=2)
				policy_optimizer.step()
			else:
				total_loss += policy_loss				

		optimiser.zero_grad()
		total_loss.backward()
		nn.utils.clip_grad_norm_(param_list, args.grad_clip_norm, norm_type=2)
		optimiser.step()
		# Store (0) observation loss (1) reward loss (2) KL loss
		losses.append([observation_loss.item(), reward_loss.item(), kl_loss.item()])
		if args.use_policy:
			if args.use_value:
				losses[-1].append(return_loss.item())
				#losses[-1].append(adv_loss.item())
			losses[-1].append(policy_loss.item())				


	# Update and plot loss metrics
	losses = tuple(zip(*losses))
	metrics['observation_loss'].append(losses[0])
	metrics['reward_loss'].append(losses[1])
	metrics['kl_loss'].append(losses[2])
	lineplot(metrics['episodes'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'observation_loss', results_dir)
	lineplot(metrics['episodes'][-len(metrics['reward_loss']):], metrics['reward_loss'], 'reward_loss', results_dir)
	lineplot(metrics['episodes'][-len(metrics['kl_loss']):], metrics['kl_loss'], 'kl_loss', results_dir)
	if args.use_value:
		metrics['value_loss'].append(losses[3])
		lineplot(metrics['episodes'][-len(metrics['value_loss']):], metrics['value_loss'], 'value_loss', results_dir)
		#metrics['advantage_loss'].append(losses[4])
		#lineplot(metrics['episodes'][-len(metrics['advantage_loss']):], metrics['advantage_loss'], 'advantage_loss', results_dir)
	if args.use_policy:
		metrics['policy_loss'].append(losses[-1])
		lineplot(metrics['episodes'][-len(metrics['policy_loss']):], metrics['policy_loss'], 'policy_loss', results_dir)
		
	
	# Data collection
	with torch.no_grad():
		observation, total_reward = env.reset(), 0
		belief, posterior_state, action = torch.zeros(1, args.belief_size, device=args.device), torch.zeros(1, args.state_size, device=args.device), torch.zeros(1, env.action_size, device=args.device)
		pbar = tqdm(range(args.max_episode_length // args.action_repeat))
		for t in pbar:
			belief, posterior_state, action, next_observation, reward, done = update_belief_and_act(args, env, planner, transition_model, encoder, belief, posterior_state, action, observation.to(device=args.device), env.action_range[0], env.action_range[1], explore=True)
			D.append(observation, action.cpu(), reward, done)
			total_reward += reward
			observation = next_observation
			if args.render:
				env.render()
			if done:
				pbar.close()
				break
		# Update and plot train reward metrics
		metrics['steps'].append(t + metrics['steps'][-1])
		metrics['episodes'].append(episode)
		metrics['train_rewards'].append(total_reward)
		lineplot(metrics['episodes'][-len(metrics['train_rewards']):], metrics['train_rewards'], 'train_rewards', results_dir)

	# Test model
	if episode % args.test_interval == 0:
		# Set models to eval mode
		transition_model.eval()
		observation_model.eval()
		reward_model.eval()
		encoder.eval()
		if args.use_policy:
			policy_net.eval()
		if args.use_value:
			value_net.eval()
			#adv_net.eval()
		# Initialise parallelised test environments
		test_envs = EnvBatcher(Env, (args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth), {}, args.test_episodes)
		
		with torch.no_grad():
			observation, total_rewards, video_frames = test_envs.reset(), np.zeros((args.test_episodes, )), []
			belief, posterior_state, action = torch.zeros(args.test_episodes, args.belief_size, device=args.device), torch.zeros(args.test_episodes, args.state_size, device=args.device), torch.zeros(args.test_episodes, env.action_size, device=args.device)
			pbar = tqdm(range(args.max_episode_length // args.action_repeat))
			for t in pbar:
				belief, posterior_state, action, next_observation, reward, done = update_belief_and_act(args, test_envs, planner, transition_model, encoder, belief, posterior_state, action, observation.to(device=args.device), env.action_range[0], env.action_range[1])
				total_rewards += reward.numpy()
				#if not args.symbolic_env:  # Collect real vs. predicted frames for video
				#  video_frames.append(make_grid(torch.cat([observation, observation_model(belief, posterior_state).cpu()], dim=3) + 0.5, nrow=5).numpy())  # Decentre
				observation = next_observation
				if done.sum().item() == args.test_episodes:
					pbar.close()
					break
		
		# Update and plot reward metrics (and write video if applicable) and save metrics
		metrics['test_episodes'].append(episode)
		metrics['test_rewards'].append(total_rewards.tolist())
		lineplot(metrics['test_episodes'], metrics['test_rewards'], 'test_rewards', results_dir)
		lineplot(np.asarray(metrics['steps'])[np.asarray(metrics['test_episodes']) - 1], metrics['test_rewards'], 'test_rewards_steps', results_dir, xaxis='step')
		#if not args.symbolic_env:
		#  episode_str = str(episode).zfill(len(str(args.episodes)))
		#  write_video(video_frames, 'test_episode_%s' % episode_str, results_dir)  # Lossy compression
		#  save_image(torch.as_tensor(video_frames[-1]), os.path.join(results_dir, 'test_episode_%s.png' % episode_str))
		torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

		# Set models to train mode
		transition_model.train()
		observation_model.train()
		reward_model.train()
		encoder.train()
		if args.use_policy:
			policy_net.train()
		if args.use_value:
			value_net.train()
			#adv_net.train()
		# Close test environments
		test_envs.close()


	# Checkpoint models
	if episode % args.checkpoint_interval == 0:
		saver = {'transition_model': transition_model.state_dict(), 'observation_model': observation_model.state_dict(), 'reward_model': reward_model.state_dict(), 'encoder': encoder.state_dict(), 'optimiser': optimiser.state_dict()} 
		if args.use_policy:
			saver['policy_net'] = policy_net.state_dict()
			if args.detach_policy:
				saver['policy_optimizer'] = policy_optimizer.state_dict()
		if args.use_value:
			saver['value_net'] = value_net.state_dict()
			#saver['adv_net']  = adv_net.state_dict()
			if args.detach_policy:
				saver['value_optimizer'] = value_optimizer.state_dict()
		torch.save(saver, os.path.join(results_dir, 'models_%d.pth' % episode))
		if args.checkpoint_experience:
			torch.save(D, os.path.join(results_dir, 'experience.pth'))  # Warning: will fail with MemoryError with large memory sizes


# Close training environment
env.close()
