import torch
import time
import os
from vmas import make_env
from moviepy import ImageSequenceClip
from tqdm import trange

class AgentTrainer:
    def __init__(
        self,
        render: bool,
        num_envs: int,
        n_steps: int,
        device: str,
        scenario,
        continuous_actions: bool,
        random_action: bool,
        **kwargs
    ):
        self.render = render
        self.num_envs = num_envs
        self.n_steps = n_steps
        self.device = device
        self.scenario = scenario
        self.continuous_actions = continuous_actions
        self.random_action = random_action
        self.env_kwargs = kwargs

    def train_agents(self, agents):

        env = make_env(
            scenario=self.scenario,
            num_envs=self.num_envs,
            device=self.device,
            continuous_actions=self.continuous_actions,
            seed=0,
            **self.env_kwargs
        )
        replay_buffer = []
        obs = env.reset()
        for s in trange(self.n_steps, desc="Training Agents"):
            actions = []
            for i, agent in enumerate(env.agents):
                agent_obs = obs[i]
                action = agents[i].choose_action(agent_obs)
                actions.append(action)
            actions = [a.cpu() if isinstance(a, torch.Tensor) else a for a in actions]
            next_obs, rews, dones, info = env.step(actions)
            for i, agent in enumerate(env.agents):
                transition = {
                    'obs': obs[i].to(self.device) if isinstance(obs[i], torch.Tensor) else torch.tensor(obs[i], dtype=torch.float32, device=self.device).unsqueeze(0),
                    'action': actions[i].to(self.device) if isinstance(actions[i], torch.Tensor) else torch.tensor(actions[i], dtype=torch.long, device=self.device).unsqueeze(0),
                    'reward': rews[i].to(self.device) if isinstance(rews[i], torch.Tensor) else torch.tensor([rews[i]], dtype=torch.float32, device=self.device),
                    'next_obs': next_obs[i].to(self.device) if isinstance(next_obs[i], torch.Tensor) else torch.tensor(next_obs[i], dtype=torch.float32, device=self.device).unsqueeze(0),
                    'done': dones[i].float().to(self.device) if isinstance(dones[i], torch.Tensor) else torch.tensor([float(dones[i])], dtype=torch.float32, device=self.device)
                }
                replay_buffer.append((i, transition))
            for i, agent in enumerate(env.agents):
                agent_transitions = [t for idx, t in replay_buffer if idx == i]
                if len(agent_transitions) > 0:
                    batch = agent_transitions[-1]
                    agents[i].update(batch)
            obs = next_obs
        return agents

    def render_gif(self, agents, render_every=1, gif_path="images/agent_trainer.gif", fps=30):
        env = make_env(
            scenario=self.scenario,
            num_envs=1,
            device=self.device,
            continuous_actions=self.continuous_actions,
            seed=0,
            **self.env_kwargs
        )
        obs = env.reset()
        frame_list = []
        for s in range(self.n_steps):
            actions = []
            for i, agent in enumerate(env.agents):
                agent_obs = obs[i]
                action = agents[i].choose_action(agent_obs, eval_mode=True)
                actions.append(action)
            actions = [a.cpu() if isinstance(a, torch.Tensor) else a for a in actions]
            next_obs, rews, dones, info = env.step(actions)
            if s % render_every == 0:
                frame = env.render(mode="rgb_array", agent_index_focus=None)
                frame_list.append(frame)
            obs = next_obs
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        clip = ImageSequenceClip(frame_list, fps=fps)
        clip.write_gif(gif_path, fps=fps)

    def make_dummy_env(self):
        return make_env(
            scenario=self.scenario,
            num_envs=1,
            device=self.device,
            continuous_actions=self.continuous_actions,
            seed=0,
            **self.env_kwargs
        )