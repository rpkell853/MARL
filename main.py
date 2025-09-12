from DQN import DQNAgent
from AgentTrainer import AgentTrainer

if __name__ == "__main__":
    scenario_name = "dispersion"
    render = True
    num_envs = 128
    n_steps = 1000
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    continuous_actions = False
    random_action = False
    n_agents = 4
    penalise_by_time = True

    # Create AgentTrainer
    trainer = AgentTrainer(
        render=render,
        num_envs=num_envs,
        n_steps=n_steps,
        device=device,
        scenario=scenario_name,
        continuous_actions=continuous_actions,
        random_action=random_action,
        n_agents=n_agents,
        penalise_by_time=penalise_by_time,
    )

    # Create DQN agents
    # We need to create a dummy environment to get obs_dim and action_dim
    dummy_env = trainer.make_dummy_env()
    obs_dim = dummy_env.observation_space[0].shape[0]
    action_dim = dummy_env.action_space[0].n
    agents = [DQNAgent(obs_dim, action_dim, device=device) for _ in range(n_agents)]

    # Train agents
    trained_agents = trainer.train_agents(agents)

    # Render gif
    for i in range(3):
        trainer.render_gif(trained_agents, render_every=2, env_seed=i, gif_path=f"images/{scenario_name}_agenttrainer.gif", fps=30)