import torch
import random

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.next_obs_buffer = []
        self.done_buffer = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device

    def add(self, transition):
        """Add a transition to the buffer"""
        self.obs_buffer.append(transition['obs'])
        self.action_buffer.append(transition['action'])
        self.reward_buffer.append(transition['reward'])
        self.next_obs_buffer.append(transition['next_obs'])
        self.done_buffer.append(transition['done'])

        if len(self.obs_buffer) > self.buffer_size:
            self.obs_buffer.pop(0)
            self.action_buffer.pop(0)
            self.reward_buffer.pop(0)
            self.next_obs_buffer.pop(0)
            self.done_buffer.pop(0)

    def sample(self):
        """Sample a batch of transitions from the buffer"""
        if len(self.obs_buffer) < self.batch_size:
            return None

        indices = random.sample(range(len(self.obs_buffer)), self.batch_size)
        
        batch = {
            'obs': torch.cat([self.obs_buffer[i] for i in indices], dim=0),
            'action': torch.cat([self.action_buffer[i] for i in indices], dim=0),
            'reward': torch.cat([self.reward_buffer[i] for i in indices], dim=0),
            'next_obs': torch.cat([self.next_obs_buffer[i] for i in indices], dim=0),
            'done': torch.cat([self.done_buffer[i].unsqueeze(0) for i in indices], dim=0)
        }
        return batch

    def __len__(self):
        return len(self.obs_buffer)