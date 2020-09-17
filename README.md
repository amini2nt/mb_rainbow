PG+
======

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

PG+: Policy Guided Planning in Latent Space or PGPLaS (PG+) improves upon PlaNet by learning a policy in the latent space and then using the policy while doing decision-time planning. This work we show an efficient way to add noise to the policy parameters that speeds up the Cross Entropy Method (CEM) search in the policy space. Our experimental results show that PG+ achieves better performance than PlaNet in all five continuous control tasks that we considered



Requirements
------------

- Python 3
- [DeepMind Control Suite](https://github.com/deepmind/dm_control) 
- [Gym](https://gym.openai.com/)
- [OpenCV Python](https://pypi.python.org/pypi/opencv-python)
- [Plotly](https://plot.ly/)
- [PyTorch](http://pytorch.org/)

To install all dependencies with Anaconda run `conda env create -f environment.yml` and use `source activate planet` to activate the environment. We used Python 3.6 to run these experiments. 


Running
------------

You can reproduce the results by running something like the following: 

python main.py --id marwil_cartpole_k_0.5_sdetach_run_1 --seed 1 --env cartpole-swingup --action-repeat 2 --use-policy True --use-value True --planner POP_P_Planner --detach-policy True --stoch-policy True --marwil-kappa 0.5

Hyperparameters:

'--id' ==> 'Experiment ID'
'--seed' ==> help='Random seed')
'--disable-cuda' ==> 'Disable CUDA')
'--env' ==> help='Gym/Control Suite environment')
'--symbolic-env' ==> 'Symbolic features')
'--max-episode-length' ==> 'Max episode length')
'--experience-size' ==> 'Experience replay size')  
'--activation-function' ==> 'Model activation function')
'--embedding-size' ==> 'Observation embedding size') 
'--hidden-size' ==> 'Hidden size')
'--belief-size' ==> 'Belief/hidden size')
'--state-size' ==> 'State/latent size')
'--action-repeat' ==> 'Action repeat')
'--action-noise' ==> 'Action noise')
'--episodes' ==> 'Total number of episodes')
'--seed-episodes' ==> 'Seed episodes')
'--collect-interval' ==> 'Collect interval')
'--batch-size' ==> 'Batch size')
'--chunk-size' ==> 'Chunk size')
'--overshooting-distance' ==> 'Latent overshooting distance/latent overshooting weight for t = 1')
'--overshooting-kl-beta' ==> help='Latent overshooting KL weight for t > 1 (0 to disable)')
'--overshooting-reward-scale' ==> help='Latent overshooting reward prediction weight for t > 1 (0 to disable)'
'--global-kl-beta' ==> 'Global KL weight (0 to disable)')
'--free-nats' ==> 'Free nats')
'--bit-depth' ==> 'Image bit depth (quantisation)')
'--learning-rate' ==> 'Learning rate') 
'--learning-rate-schedule' ==> 'Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)' 
'--adam-epsilon' ==> 'Adam optimiser epsilon value') 
'--grad-clip-norm' ==> 'Gradient clipping norm')
'--planning-horizon' ==> 'Planning horizon distance')
'--optimisation-iters' ==> 'Planning optimisation iterations')
'--candidates' ==> 'Candidate samples per iteration')
'--top-candidates' ==> 'Number of top candidates to fit')
'--test', action='store_true', help='Test only')
'--test-interval' ==> 'Test interval (episodes)')
'--test-episodes' ==> 'Number of test episodes')
'--checkpoint-interval' ==> 'Checkpoint interval (episodes)')
'--checkpoint-experience' ==> 'Checkpoint experience replay')
'--models' ==> 'Load model checkpoint')
'--experience-replay' ==> 'Load experience replay')
'--render' ==> 'Render environment')

#New set of hyper-parameters
-----
'--initial-sigma' ==> 'Initial sigma value for CEM')
'--use-policy' ==> 'Use a policy network')
'--stoch-policy' ==> 'Use a stochastic policy')
'--detach-policy' ==> 'no updates from policy to model')
'--use-value' ==> 'Use a value network')
'--planner' ==> 'Type of planner')
'--policy-reduce' ==> 'policy loss reduction')
'--mppi-gamma' ==> 'MPPI gamma')
'--mppi-beta' ==> 'MPPI beta')
'--marwil-kappa' ==> 'kappa for marwil')
'--res-dir' ==> 'directory location')
'--policy-lr' ==> 'policy Learning rate')
'--value-lr' ==> 'policy Learning rate')
'--policy-adam' ==> 'policy Learning rate')
'--value-adam' ==> 'policy Learning rate')

Links
-----
- [The PlaNet implementation used in this work](https://github.com/Kaixhin/PlaNet)

- [Introducing PlaNet: A Deep Planning Network for Reinforcement Learning](https://ai.googleblog.com/2019/02/introducing-planet-deep-planning.html)
- [google-research/planet](https://github.com/google-research/planet)

Acknowledgements
----------------

- [@danijar](https://github.com/danijar) for [google-research/planet](https://github.com/google-research/planet) and [help reproducing results](https://github.com/google-research/planet/issues/28)
- [@sg2](https://github.com/sg2) for [running experiments](https://github.com/Kaixhin/PlaNet/issues/9)
- [@Kaixhin](https://github.com/Kaixhin) for reproducing the PlaNet experiments. 

References
----------

[1] [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551) 
[2] [Exploring Model-based Planning with Policy Networks](https://arxiv.org/abs/1906.08649) 
[3] [Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models](https://arxiv.org/abs/1805.12114)
