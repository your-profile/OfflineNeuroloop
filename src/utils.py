import gymnasium
from copy import deepcopy as dc
from src.networks.DQN import DQN
from src.networks.DDPG import DDPG
from src.envs.lunar_lander import LunarLander
from src.envs.flappy_bird import FlappyBirdEnv
from src.seed_utils import set_global_seed
import torch

def make_fetch_env(max_episode_steps=50, mujoco_version: int = 4, verbose: bool = False, render_mode: str = "rgb_array"):
    """
    OpenAI Gymnasium Fetch Pick and Place
    """
    import gymnasium_robotics

    gymnasium.register_envs(gymnasium_robotics)

    return gymnasium.make("FetchPickAndPlace-v4", render_mode=render_mode, max_episode_steps=max_episode_steps)
    

def get_percentiles(domain: str):
    if domain[0].lower() == "l":
        return (5.2, -1.5, -2.9) #0.9, 0.6, 0.3
    elif domain[0].lower() == "f":
        return (1.0, 0.1, -1.0) #0.99, 0.5, 0.01
    elif domain[0].lower() == "r":
        return (0.0, -0.5, -1.0) #0.99, 0.5, 0.01
    else:
        raise Exception(f"Invalid domain: {domain}. Try: Lunar Lander, Flappy Bird, Robot")

def load_domain(env: str, steps: int = None):
    if env[0].lower() == "l":
        env = LunarLander()
    elif env[0].lower() == "f":
        env = FlappyBirdEnv(score_limit=100)
    elif env[0].lower() == "r":
        env = make_fetch_env(max_episode_steps=steps)
    else:
        raise Exception(f"Invalid domain: {env}. Try: Lunar Lander, Flappy Bird, Robot")

    return env

def load_pretrained_agent(agent: DQN | DDPG, filename:str,pretrained_success_rate: float, algorithm: str, space=(11, 4), verbose: bool = False):
    """ Loading pretrained agents for Lunar Lander, Flappy Bird, and Robot """

    if algorithm == "DQN":
        if space[0] == 11: # Lunar Lander
            if verbose:
                print(filename+"lunar/"+"LPolicy"+str(int(pretrained_success_rate)))

            agent = agent.load_model(filename = filename+"src/policies/lunar/"+"LPolicy"+str(int(pretrained_success_rate)))
        elif space[0] == 12:
            if verbose:
                print(filename+"flappy/"+"FPolicy"+str(int(pretrained_success_rate)))
            agent = agent.load_model(filename = filename+"src/policies/flappy/"+"FPolicy"+str(int(pretrained_success_rate)))
        
    if algorithm == "DDPG": # Robot
        if verbose:
            print(filename+"robot/"+"FetchPolicy"+str(int(pretrained_success_rate)))
        agent = agent.load_model(filename = filename+"src/policies/robot/"+"FetchPolicy"+str(int(pretrained_success_rate)) + ".pth")
        print("Loaded DDPG agent from: " + filename+"src/policies/robot/"+"FetchPolicy"+str(int(pretrained_success_rate)) + ".pth")
    return agent

def load_agent(algorithm: str, buffer_type: str, filename:str, space=(11, 4), pretrained_success_rate: float = 0.0, seed: int | None = None, verbose: bool = False):
    
    """ Loading DQN or DDPG agents for Lunar Lander, Flappy Bird, and Robot """

    agent = None

    if seed is not None:
        set_global_seed(seed)

    if algorithm == "DQN":
        if space[0] == 11: # Lunar Lander
            hidden_layer_size = 128
            agent = DQN(
                n_observations=space[0],
                n_actions=space[1],
                batch_size=128,
                lr=1e-3,
                gamma=0.99,
                mem_size=100000,
                learn_step=5,
                tau=0.005,
                buffer_type=buffer_type,
                hidden_layer_size=hidden_layer_size,
                seed=seed,
                verbose=verbose
            )
        else: # Flappy Bird
            hidden_layer_size = 128
            agent = DQN(
                n_observations=space[0],
                n_actions=space[1],
                batch_size=128,
                lr=1e-3,
                gamma=0.99,
                mem_size=100000,
                learn_step=5,
                tau=0.005,
                buffer_type=buffer_type,
                hidden_layer_size=hidden_layer_size,
                seed=seed,
                verbose=verbose
            )

        if verbose:
            print(f"Loading DQN agent with {space[0]} observations and {space[1]} actions")

    elif algorithm == "DDPG": # Robot
        inner = make_fetch_env(max_episode_steps=50, verbose = verbose)
        agent = load_ddpg_agent(inner, buffer_type, seed=seed, verbose = verbose)

    if pretrained_success_rate > 0.0:
        return load_pretrained_agent(agent=agent, filename=filename, pretrained_success_rate=pretrained_success_rate, algorithm=algorithm, space=space, verbose = verbose)

    return agent


def load_ddpg_agent(env, buffer_type: str, seed: int | None = None, verbose: bool = False, pretrained_success_rate: float = 0.0):
    """Build DDPG + HER + Prioritization for Robot Pick and Place """

    memory_size = 7e+5
    batch_size = 256
    actor_lr = 1e-3
    critic_lr = 1e-3
    gamma = 0.98
    tau = 0.05
    k_future = 4

    test_env = gymnasium.make("FetchPickAndPlace-v4")
    state_shape = test_env.observation_space.spaces["observation"].shape
    n_actions = test_env.action_space.shape[0]
    n_goals = test_env.observation_space.spaces["desired_goal"].shape[0]
    action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]


    agent = DDPG(n_states=state_shape,
            n_actions=n_actions,
            n_goals=n_goals,
            action_bounds=action_bounds,
            capacity=memory_size,
            action_size=n_actions,
            batch_size=batch_size,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            tau=tau,
            k_future=k_future,
            env=dc(env),
            seed=seed,
            verbose=verbose)
            
    if pretrained_success_rate > 0.0:
        return load_pretrained_agent(agent=agent, pretrained_success_rate=pretrained_success_rate, algorithm="DDPG", space=(12, 2), verbose = verbose)
    
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
