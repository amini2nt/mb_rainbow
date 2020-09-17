PG+
======

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

PG+: Policy Guided Planning in Latent Space or PGPLaS (PG+) improves upon PlaNet by learning a policy in the latent space and then using the policy while doing decision-time planning. In this work, we show an efficient way to add noise to the policy parameters that speeds up the Cross Entropy Method (CEM) search in the policy space. Our experimental results show that PG+ achieves better performance than PlaNet in all five continuous control tasks that we considered



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

Following command will reproduce the results for Cheetah Run experiment: 

python main.py --id cheetah2_sigma_0.1_run_1 --seed 1 --initial-sigma 0.1 --use-policy True --stoch-policy True --detach-policy True --use-value True --planner POP_P_Planner --marwil-kappa 0.7 --res-dir cheetah2 --policy-lr 1e-3 --policy-adam 1e-8 --value-lr 2e-3 --value-adam 1e-8 --learning-rate 6e-4 --adam-epsilon 1e-3 --env cheetah-run --action-repeat 2 



Hyperparameters
------------

'--initial-sigma' ==> 'Initial sigma value for CEM'

'--use-policy' ==> 'Use a policy network'

'--stoch-policy' ==> 'Use a stochastic policy'

'--detach-policy' ==> 'no updates from policy to model'

'--use-value' ==> 'Use a value network'

'--planner' ==> 'Type of planner'

'--policy-reduce' ==> 'policy loss reduction'

'--marwil-kappa' ==> 'kappa for marwil'

'--res-dir' ==> 'directory location'

'--policy-lr' ==> 'policy Learning rate'

'--value-lr' ==> 'value network Learning rate'


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
