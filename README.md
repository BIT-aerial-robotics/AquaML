# AquaML

## Features

1. Support reinforcement learning, generative learning algorithm.
2. Support reinforcement learning training with recurrent neural networks.
3. Support multi-thread sampling and parameters tuning.
4. Support high performance computer(HPC)
5.  Data communication has almost zero latency when running on a single machine.

## Install

## Tutorials

### Build neural network model for reinforcement learning

### Build new reinforcement algorithm

## Change logs

#### v1.1 

1. unify ```MPIRuner``` API.
2. Add ``com`` package, it contains all base class.
3. ``save_data`` and ``load_data`` are created for supervised learning and expert learning. 
4. Gradually convert our framework to next generation like HPC-v0.1.
5. The following algos just use ``DataCollector``(support all type of algo) instead of ``DataManeger``.
6. Add plot tools for rosbag, paper.

#### v2.0

1. split optimize thread and worker thread.
2. add soft actor critic.



## Requirement

seaborn<=0.9
