import gymnasium


def make_fetch_env(max_episode_steps=None, mujoco_version: int = 2):
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
        "FetchPickAndPlace-v3",
        "FetchPickAndPlace-v4",
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
        from src.envs.lunar_lander import LunarLander
        env = LunarLander()
    elif env[0].lower() == "f":
        from src.envs.flappy_bird import FlappyBirdEnv
        env = FlappyBirdEnv()
    elif env[0].lower() == "r":
        env = make_fetch_env(max_episode_steps=steps, mujoco_version=2)
    else:
        Exception("Incorrect domain key received. Domains are: \n lunar_lander \n flappy_bird \n robot")

    return env

def load_agent(algorithm: str, buffer_type: str, space=(11, 4)):
    agent = None
    if algorithm == "DQN":
        from src.networks.DQN import DQN

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
        )

    elif algorithm == "DDPG":
        inner = make_fetch_env(max_episode_steps=50, mujoco_version=2)
        agent = load_ddpg_agent(inner, buffer_type)

    return agent


def load_ddpg_agent(env, buffer_type: str):
    """Build DDPG + HER replay for an existing Fetch env (same obs/action space as training)."""
    from src.networks.DDPG import DDPG, MemoryBuffer, PrioritizedMemoryBuffer

    obs, _ = env.reset()
    n_states = (obs["observation"].shape[0],)
    n_goals = obs["desired_goal"].shape[0]
    n_actions = env.action_space.shape[0]
    action_bounds = (env.action_space.low, env.action_space.high)
    TAU = 0.005
    ACTOR_LR = 1e-3
    CRITIC_LR = 1e-3
    GAMMA = 0.98
    DECAY_RATE = 0.995

    if buffer_type == "PER":
        ram = PrioritizedMemoryBuffer(
            size=100000,
            k_future=4,
            env=env.unwrapped,
            alpha=0.6,
        )
    else:
        ram = MemoryBuffer(size=100000, k_future=4, env=env.unwrapped)

    return DDPG(
        n_states=n_states,
        n_actions=n_actions,
        n_goals=n_goals,
        action_bounds=action_bounds,
        capacity=100000,
        env=env.unwrapped,
        k_future=4,
        batch_size=128,
        ram=ram,
        action_size=n_actions,
        tau=TAU,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        gamma=GAMMA,
        decay_rate=DECAY_RATE,
    )

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

