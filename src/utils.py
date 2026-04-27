import gymnasium
from copy import deepcopy as dc
from src.networks.DQN import DQN
from src.networks.DDPG import DDPG
from src.networks.DDPG_PER import DDPG as DDPG_HER
from src.envs.lunar_lander import LunarLander
from src.envs.flappy_bird import FlappyBirdEnv
import torch

def make_fetch_env(max_episode_steps=50, mujoco_version: int = 4, verbose: bool = False):
    """
    OpenAI Gym / Gymnasium Fetch Pick and Place. Prefer v2 when registered;
    fall back to v3/v4 if the installed gymnasium-robotics build omits v2.
    """
    import gymnasium_robotics

    gymnasium.register_envs(gymnasium_robotics)
    kwargs = {}
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    for vid in (
        f"FetchPickAndPlace-v{mujoco_version}",
        "FetchPickAndPlace-v4",
        "FetchPickAndPlace-v3",
        "FetchPickAndPlace-v2",
    ):
        try:
            return gymnasium.make(vid, **kwargs)
        except gymnasium.error.NameNotFound:
            continue
    raise gymnasium.error.NameNotFound(
        "No FetchPickAndPlace env found (tried v2/v3/v4). Install gymnasium-robotics."
    )


def load_domain(env: str, steps: int = None):
    if env[0].lower() == "l":
        env = LunarLander()
    elif env[0].lower() == "f":
        env = FlappyBirdEnv(score_limit=30)
    elif env[0].lower() == "r":
        env = make_fetch_env(max_episode_steps=steps, mujoco_version=4)
    else:
        Exception("Incorrect domain key received. Domains are: \n lunar_lander \n flappy_bird \n robot")

    return env

def load_pretrained_agent(agent: DQN | DDPG | DDPG_HER, pretrained_success_rate: float, algorithm: str, space=(11, 4), filename: str = "/cluster/home/mbrowe02/OfflineNeuroloop/src/policies/", verbose: bool = False):
    
    if algorithm == "DQN":
        if space[0] == 11:
            if verbose:
                print(filename+"lunar/"+"LPolicy"+str(int(pretrained_success_rate)))

            agent = agent.load_model(filename = filename+"lunar/"+"LPolicy"+str(int(pretrained_success_rate)))
        elif space[0] == 12:
            if verbose:
                print(filename+"flappy/"+"FPolicy"+str(int(pretrained_success_rate)))
            agent = agent.load_model(filename = filename+"flappy/"+"FPolicy"+str(int(pretrained_success_rate)))
        
    if algorithm == "DDPG":
        if verbose:
            print(filename+"robot/"+"FetchPolicy"+str(int(pretrained_success_rate)))
        agent = agent.load_model(filename = filename+"robot/"+"FetchPolicy"+str(int(pretrained_success_rate)) + ".pth")
        print("Loaded DDPG agent from: " + filename+"robot/"+"FetchPolicy"+str(int(pretrained_success_rate)) + ".pth")
    return agent

def load_agent(algorithm: str, buffer_type: str, space=(11, 4), pretrained_success_rate: float = 0.0, verbose: bool = False):
    agent = None

    if algorithm == "DQN":
        if space[0] == 11:
            hidden_layer_size = 256
        else:
            hidden_layer_size = 192

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
            verbose=verbose
        )

    elif algorithm == "DDPG":
        inner = make_fetch_env(max_episode_steps=50, mujoco_version=4, verbose = verbose)
        agent = load_ddpg_agent(inner, buffer_type, verbose = verbose)

    if pretrained_success_rate > 0.0:
        return load_pretrained_agent(agent=agent, pretrained_success_rate=pretrained_success_rate, algorithm=algorithm, space=space, verbose = verbose)

    return agent


def load_ddpg_agent(env, buffer_type: str, verbose: bool = False, pretrained_success_rate: float = 0.0):
    """Build DDPG + HER replay for an existing Fetch env (same obs/action space as training)."""

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


    if buffer_type == "PER":
        agent = DDPG_HER(n_states=state_shape,
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
                verbose=verbose)
        if pretrained_success_rate > 0.0:
            return load_pretrained_agent(agent=agent, pretrained_success_rate=pretrained_success_rate, algorithm="DDPG", space=(12, 2), verbose = verbose)
    else:

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
