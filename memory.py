import numpy as np
import torch
from env import postprocess_observation, preprocess_observation_


class ExperienceReplay():
  def __init__(self, size, symbolic_env, observation_size, action_size, bit_depth, device):
    self.device = device
    self.symbolic_env = symbolic_env
    self.size = size
    self.observations = np.empty((size, observation_size) if symbolic_env else (size, 3, 64, 64), dtype=np.float32 if symbolic_env else np.uint8)
    self.actions = np.empty((size, action_size), dtype=np.float32)
    self.rewards = np.empty((size, ), dtype=np.float32) 
    self.nonterminals = np.empty((size, 1), dtype=np.float32)
    self.idx = 0
    self.full = False  # Tracks if memory has been filled/all slots are valid
    self.steps, self.episodes = 0, 0  # Tracks how much experience has been used in total
    self.bit_depth = bit_depth

  def append(self, observation, action, reward, done):
    if self.symbolic_env:
      self.observations[self.idx] = observation.numpy()
    else:
      self.observations[self.idx] = postprocess_observation(observation.numpy(), self.bit_depth)  # Decentre and discretise visual observations (to save memory)
    self.actions[self.idx] = action.numpy()
    self.rewards[self.idx] = reward
    self.nonterminals[self.idx] = not done
    self.idx = (self.idx + 1) % self.size
    self.full = self.full or self.idx == 0
    self.steps, self.episodes = self.steps + 1, self.episodes + (1 if done else 0)

  # Returns an index for a valid single sequence chunk uniformly sampled from the memory
  def _sample_idx(self, L):
    valid_idx = False
    while not valid_idx:
      idx = np.random.randint(0, self.size if self.full else self.idx - L)
      idxs = np.arange(idx, idx + L) % self.size
      valid_idx = not self.idx in idxs[1:]  # Make sure data does not cross the memory index
    return idxs

  def _retrieve_batch(self, idxs, n, L):
    vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
    observations = torch.as_tensor(self.observations[vec_idxs].astype(np.float32))
    if not self.symbolic_env:
      preprocess_observation_(observations, self.bit_depth)  # Undo discretisation for visual observations
    return observations.reshape(L, n, *observations.shape[1:]), self.actions[vec_idxs].reshape(L, n, -1), self.rewards[vec_idxs].reshape(L, n), self.nonterminals[vec_idxs].reshape(L, n, 1)

  # Returns a batch of sequence chunks uniformly sampled from the memory
  def sample(self, n, L):
    batch = self._retrieve_batch(np.asarray([self._sample_idx(L) for _ in range(n)]), n, L)
    return [torch.as_tensor(item).to(device=self.device) for item in batch]


class ExperienceReplayv2():
  def __init__(self, size, symbolic_env, observation_size, action_size, bit_depth, device):
    self.device = device
    self.symbolic_env = symbolic_env
    self.size = size
    self.observations = np.empty((size, observation_size) if symbolic_env else (size, 3, 64, 64), dtype=np.float32 if symbolic_env else np.uint8)
    self.actions = np.empty((size, action_size), dtype=np.float32)
    self.rewards = np.empty((size, ), dtype=np.float32)
    self.returns = np.empty((size, ), dtype=np.float32)  
    self.nonterminals = np.empty((size, 1), dtype=np.float32)
    self.idx = 0
    self.full = False  # Tracks if memory has been filled/all slots are valid
    self.steps, self.episodes = 0, 0  # Tracks how much experience has been used in total
    self.bit_depth = bit_depth
    self.episode_ids = []
    self.gamma = 0.99
  
  def update_returns(self):

    self.returns[self.episode_ids[-1]] = self.rewards[self.episode_ids[-1]]

    for i in range(1, len(self.episode_ids)):
      idx = len(self.episode_ids) - 1 - i
      self.returns[self.episode_ids[idx]] = self.rewards[self.episode_ids[idx]] + self.gamma * self.returns[self.episode_ids[idx+1]]

    self.episode_ids = []


  def append(self, observation, action, reward, done):
    if self.symbolic_env:
      self.observations[self.idx] = observation.numpy()
    else:
      self.observations[self.idx] = postprocess_observation(observation.numpy(), self.bit_depth)  # Decentre and discretise visual observations (to save memory)
    self.actions[self.idx] = action.numpy()
    self.rewards[self.idx] = reward
    self.nonterminals[self.idx] = not done

    self.episode_ids.append(self.idx)
    if done:
      self.update_returns()

    self.idx = (self.idx + 1) % self.size
    self.full = self.full or self.idx == 0
    self.steps, self.episodes = self.steps + 1, self.episodes + (1 if done else 0)

  # Returns an index for a valid single sequence chunk uniformly sampled from the memory
  def _sample_idx(self, L):
    valid_idx = False
    while not valid_idx:
      idx = np.random.randint(0, self.size if self.full else self.idx - L)
      idxs = np.arange(idx, idx + L) % self.size
      #valid_idx = not self.idx in idxs[1:]  # Make sure data does not cross the memory index
      if len(list(set(self.episode_ids) & set(idxs))) == 0 and self.idx not in idxs[1:]:
        valid_idx = True
    return idxs

  def _retrieve_batch(self, idxs, n, L):
    vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
    observations = torch.as_tensor(self.observations[vec_idxs].astype(np.float32))
    if not self.symbolic_env:
      preprocess_observation_(observations, self.bit_depth)  # Undo discretisation for visual observations
    return observations.reshape(L, n, *observations.shape[1:]), self.actions[vec_idxs].reshape(L, n, -1), self.rewards[vec_idxs].reshape(L, n), self.nonterminals[vec_idxs].reshape(L, n, 1), self.returns[vec_idxs].reshape(L, n)

  # Returns a batch of sequence chunks uniformly sampled from the memory
  def sample(self, n, L):
    batch = self._retrieve_batch(np.asarray([self._sample_idx(L) for _ in range(n)]), n, L)
    return [torch.as_tensor(item).to(device=self.device) for item in batch]
