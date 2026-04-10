TO FRIKIN DO:
- Connect yaml configurations to actual experiment code
- Add robot code and finalize all conditions
- Configure baseline experiments for all environments
- Add results to logging
- Add agent and classifier parameters
- Write some testing files to make sure things are running as expected
- Create figures from logged results (ablations, main experiments)


```
project/
├── configs/
│   ├── base.yaml
│   ├── domains/
│   │   ├── robot.yaml
│   │   ├── lunar_lander.yaml
│   │   └── flappy_bird.yaml
│   └── ablations/
│       ├── noise_sweep.yaml
│       ├── smoothing_sweep.yaml
│       └── credit_sweep.yaml
├── src/
│   ├── neural/
│   │   ├── preprocessing.py      # smoothing, noise injection
│   │   ├── credit.py             # credit assignment methods
│   │   └── conditions.py         # reward_shaping, lr_modulation, replay_priority
│   ├── models/
│   │   ├── binary.py             # binary classifier MLP
│   │   ├── ternary.py            # ternary classifier MLP
│   │   └── continuous.py         # continuous output MLP
│   ├── envs/
│   │   ├── robot.py              # environments
│   │   ├── lunar_lander.py
│   │   └── flappy_bird.py
│   └── logging.py
├── train.py
└── run_experiments.py
```