def load_domain(env: str = "None"):
    if env == "lunar_lander":
        from src.envs.lunar_lander import LunarLander
        env = LunarLander()
    elif env == "flappy_bird":
        from src.envs.flappy_bird import FlappyBirdEnv
        env = FlappyBirdEnv()
    elif env == "robot":
        print("TODO")
    else:
        Exception("Incorrect domain key received. Domains are: \n lunar_lander \n flappy_bird \n robot")

    return env

def load_agent(algorithm: str, buffer_type: str, space = (11, 4)):
    if algorithm == "DQN":
        from src.networks.DQN import DQN
        agent = DQN(
            n_observations = space[0],
            n_actions = space[1],
            batch_size = 128,
            lr = 1e-3,
            gamma = 0.99,
            mem_size = 100000,
            learn_step = 5,
            tau = 0.005,
            buffer_type=buffer_type) 
    return agent
