from vmas.simulator.scenario import BaseScenario
from typing import Union
import time
import torch
from vmas import make_env
from vmas.simulator.core import Agent
import numpy as np
from DQN import DQNAgent

def _get_deterministic_action(agent: Agent, continuous: bool, env):
    if continuous:
        action = -agent.action.u_range_tensor.expand(env.batch_dim, agent.action_size)
    else:
        action = (
            torch.tensor([1], device=env.device, dtype=torch.long)
            .unsqueeze(-1)
            .expand(env.batch_dim, 1)
        )
    return action.clone()

def use_vmas_env(
    render: bool,
    num_envs: int,
    n_steps: int,
    device: str,
    scenario: Union[str, BaseScenario],
    continuous_actions: bool,
    random_action: bool,
    **kwargs
):
    """Example function to use a vmas environment.
    
    This is a simplification of the function in `vmas.examples.use_vmas_env.py`.

    Args:
        continuous_actions (bool): Whether the agents have continuous or discrete actions
        scenario (str): Name of scenario
        device (str): Torch device to use
        render (bool): Whether to render the scenario
        num_envs (int): Number of vectorized environments
        n_steps (int): Number of steps before returning done
        random_action (bool): Use random actions or have all agents perform the down action

    """

    scenario_name = scenario if isinstance(scenario,str) else scenario.__class__.__name__

    env = make_env(
        scenario=scenario,
        num_envs=num_envs,
        device=device,
        continuous_actions=continuous_actions,
        seed=0,
        # Environment specific variables
        **kwargs
    )

    # --- DQN Setup ---
    # Assume all agents have the same obs/action space for simplicity
    obs_dim = env.observation_space[0].shape[0]  # Assuming obs is a 1D array
    action_dim = env.action_space[0].n
    dqn_agents = [DQNAgent(obs_dim, action_dim, device=device) for _ in env.agents]
    replay_buffer = []

    frame_list = []  # For creating a gif
    init_time = time.time()
    step = 0

    obs = env.reset()
    for s in range(n_steps):
        step += 1
        print(f"Step {step}")

        actions = []
        for i, agent in enumerate(env.agents):
            agent_obs = obs[i]
            action = dqn_agents[i].choose_action(agent_obs)
            actions.append(action)
        # Convert actions to expected format for env.step
        actions = [a.cpu() if isinstance(a, torch.Tensor) else a for a in actions]

        next_obs, rews, dones, info = env.step(actions)

        # Store transitions in replay buffer
        for i, agent in enumerate(env.agents):
            transition = {
                'obs': obs[i].to(device) if isinstance(obs[i], torch.Tensor) else torch.tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0),
                'action': actions[i].to(device) if isinstance(actions[i], torch.Tensor) else torch.tensor(actions[i], dtype=torch.long, device=device).unsqueeze(0),
                'reward': rews[i].to(device) if isinstance(rews[i], torch.Tensor) else torch.tensor([rews[i]], dtype=torch.float32, device=device),
                'next_obs': next_obs[i].to(device) if isinstance(next_obs[i], torch.Tensor) else torch.tensor(next_obs[i], dtype=torch.float32, device=device).unsqueeze(0),
                'done': dones[i].float().to(device) if isinstance(dones[i], torch.Tensor) else torch.tensor([float(dones[i])], dtype=torch.float32, device=device)
            }
            replay_buffer.append((i, transition))

        # DQN Training step (simple online, not batch)
        for i, agent in enumerate(env.agents):
            # Sample recent transition for online update
            agent_transitions = [t for idx, t in replay_buffer if idx == i]
            if len(agent_transitions) > 0:
                batch = agent_transitions[-1]
                dqn_agents[i].update(batch)

        obs = next_obs

        if render:
            frame = env.render(
                mode="rgb_array",
                agent_index_focus=None,  # Can give the camera an agent index to focus on
            )
            frame_list.append(frame)

    total_time = time.time() - init_time
    print(
        f"It took: {total_time}s for {n_steps} steps of {num_envs} parallel environments on device {device} "
        f"for {scenario_name} scenario."
    )

    if render:
        from moviepy import ImageSequenceClip
        import os
        fps = 30
        os.makedirs("images", exist_ok=True)
        clip = ImageSequenceClip(frame_list, fps=fps)
        clip.write_gif(os.path.join("images", f"{scenario_name}.gif"), fps=fps)

if __name__ == "__main__":
    scenario_name="dispersion"
    use_vmas_env(
        scenario=scenario_name,
        render=True,
        num_envs=32,
        n_steps=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
        continuous_actions=False,
        random_action=False,
        # Environment specific variables
        n_agents=4,
    )