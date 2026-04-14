import gymnasium


def load_domain(env: str, steps: int = None):
    if env[0].lower() == "l":
        from src.envs.lunar_lander import LunarLander
        env = LunarLander()
    elif env[0].lower() == "f":
        from src.envs.flappy_bird import FlappyBirdEnv
        env = FlappyBirdEnv()
    elif env[0].lower() == "r":
        env = gymnasium.make('FetchPickAndPlace-v4', max_episode_steps=steps)
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

def get_conditions(domain, task: str, verbose = False):

    domain_letter = str(domain[0]).upper()
    condition_list = []

    if task.lower() == "passive":
        condition_list.append(str(domain_letter) + "W")
        assert(len(condition_list) == 1)

    
    if task.lower() == "active":
        condition_list.append(str(domain_letter) + "P")
        assert(len(condition_list) == 1)

    
    if task.lower() == "pooled":
        condition_list.append(str(domain_letter) + "W")
        condition_list.append(str(domain_letter) + "P")
        assert(len(condition_list) == 2)

    if verbose: print(f"Condition List: {condition_list}")

    return condition_list

def get_labels(granularity, domain, reward):

    #optimality ground truths (lunar lander)

    if reward > 90.0:
        return 0
    elif reward > 17.0:
        return 1
    else:
        return 2

