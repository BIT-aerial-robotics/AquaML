# Worker设计思路

该部分首先需要对环境进行相应的规定。

所有数据最终提交到data_module中数据shape为(num_envs, steps, dims)，其中num_envs为环境的数量，steps为每个环境的步数，dims为每个环境的维度。

所有类型的auto_step,会自动执行reset，然后执行step，直到done为True。
在auto_step中会返回next_obs和compute_obs，其中next_obs为下一个状态，compute_obs为下一时刻可以用于计算的状态。

输入到worker中的数据shape为（num_envs, dims），其中num_envs为环境的数量，dims为每个环境的维度。所有worker输出的数据shape为（num_envs, steps, dims），其中num_envs为环境的数量，steps为每个环境的步数，dims为每个环境的维度。


## env的设计

在新的env设计中，完全符合新的gymnasium标准，action的shape为（dims，），observation的shape为（dims，），reward为一个标量。

## vectorized env

该部分数据返回的shape为（num_envs, dims），其中num_envs为环境的数量，dims为每个环境的维度。

## RLWorker

RLWorker会通过Collector收集数据，将数据格式打包成（num_envs, steps, dims）的格式。