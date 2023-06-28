from abc import ABC, abstractmethod
from AquaML.core.RLToolKit import VecCollector, VecMDPCollector


class BaseWorker(ABC):
    """
    The base class of worker.
    
    All the worker should inherit this class.
    
    reward_info should be a specified.

    obs_info should be a specified.
    """


class RLAgentWorker(BaseWorker):
    """
    单智能体强化学习数据采样器。

    多线程的时候每个线程自动拥有一个agent。
    """

    def __init__(self,
                 max_steps,
                 ):

        # 参数设置
        self.max_steps = max_steps

        # 运行过程参数
        self.reset_flag = True
        self.obs = None
        self.episode_step_count = 0

    def step(self, env, agent, collector, step, rollout_steps):
        """
        采样一次数据。
        """

        if self.reset_flag:
            self.obs, flag = env.reset()
            self.reset_flag = False

            # TODO: aditional reset 这个地方作为未来的接口
            self.episode_step_count = 0

        action_dict = agent.get_action(self.obs)

        obs_, reward, done, info = env.step(action_dict)

        self.episode_step_count += 1

        if self.episode_step_count >= self.max_steps:
            done = True

        if step >= rollout_steps - 1:
            done = True

        if done:
            self.reset_flag = True
            mask = 0
        else:
            mask = 1

        collector.store_data(
            obs=self.obs,
            action=action_dict,
            reward=reward,
            next_obs=obs_,
            mask=mask
        )

        self.obs = obs_

    def roll(self, env, agent, rollout_steps, collector):

        collector.reset()
        self.reset_flag = True

        for step in range(rollout_steps):
            self.step(
                env=env,
                agent=agent,
                collector=collector,
                step=step,
                rollout_steps=rollout_steps
            )


class Evaluator(BaseWorker):
    """
    单智能体强化学习数据采样器。
    """

    def __init__(self, ):

        # 运行过程参数
        self.reset_flag = True
        self.obs = None
        self.episode_step_count = 0

    def step(self, env, agent, collector):
        """
        采样一次数据。
        """

        action_dict = agent.get_action(self.obs, test_flag=True)

        obs_, reward, done, info = env.step(action_dict)

        self.episode_step_count += 1

        if done:
            self.reset_flag = True
            mask = 0
        else:
            mask = 1

        # TODO: 数据存储策略设计

        self.obs = obs_

        collector.store_data(
            reward=reward,
        )

    def roll(self, env, agent, episode_length, episodes, collector):

        collector.reset()

        for _ in range(episodes):

            self.obs, flag = env.reset()

            for _ in range(episode_length):
                self.step(
                    env=env,
                    agent=agent,
                    collector=collector
                )

            collector.summary_episode()


class RLVectorEnvWorker(BaseWorker):
    """
    vectorized environment worker.

    The agent only runs in main process.

    # TODO: cancle max_steps
    """

    def __init__(self,
                 max_steps,
                 communicator,
                 optimize_enable,
                 sample_enable,
                 vec_env,
                 action_names: list or tuple,
                 obs_names: list or tuple,
                 reward_names: list or tuple
                 ):

        self.obs = None
        self.communicator = communicator

        # 参数设置
        self.max_steps = max_steps

        self.thread_level = self.communicator.get_level()

        self.optimize_enable = optimize_enable
        self.sample_enable = sample_enable

        self.vec_env = vec_env

        self.action_names = action_names
        self.obs_names = obs_names
        self.reward_names = reward_names

        if self.thread_level == 0:
            # main process
            self.start_index = 0
            self.end_index = self.communicator.get_data_pool_size()

        else:
            # sub process
            self.start_index = self.communicator.data_pool_start_index
            self.end_index = self.communicator.data_pool_end_index

        self.initial_flag = True

        self.vec_MDP_collector = VecMDPCollector(
            obs_names=self.obs_names,
            reward_names=self.reward_names,
            action_names=self.action_names
        )

    def step(self, agent):

        if self.optimize_enable:

            # format of data: (mxn, ...)
            # m: number of threads, n: number envs in each thread

            actions = agent.get_action(self.obs)

            # push to data pool
            self.communicator.store_data_dict(
                agent_name=agent.name,
                data_dict=actions,
                start_index=self.start_index,
                end_index=self.end_index
            )

        self.communicator.Barrier()

        if self.sample_enable:
            # TODO: 这里的代码需要优化
            # get action from data pool
            actions = self.communicator.get_pointed_data_pool_dict(
                agent_name=agent.name,
                data_name=self.action_names,
                start_index=self.start_index,
                end_index=self.end_index
            )

            # step
            next_obs, reward, done = self.vec_env.step(actions)

            self.communicator.store_data_dict(
                agent_name=agent.name,
                data_dict=next_obs,
                start_index=self.start_index,
                end_index=self.end_index
            )

            self.communicator.store_data_dict(
                agent_name=agent.name,
                data_dict=reward,
                start_index=self.start_index,
                end_index=self.end_index
            )

            self.communicator.store_data_dict(
                agent_name=agent.name,
                data_dict={'mask': done},
                start_index=self.start_index,
                end_index=self.end_index
            )

        self.communicator.Barrier()

        if self.optimize_enable:
            # store data to MDP collector
            next_obs = self.communicator.get_pointed_data_pool_dict(
                agent_name=agent.name,
                data_name=self.obs_names,
                start_index=self.start_index,
                end_index=self.end_index
            )

            reward = self.communicator.get_pointed_data_pool_dict(
                agent_name=agent.name,
                data_name=self.reward_names,
                start_index=self.start_index,
                end_index=self.end_index
            )

            mask = self.communicator.get_pointed_data_pool_dict(
                agent_name=agent.name,
                data_name=['mask'],
                start_index=self.start_index,
                end_index=self.end_index
            )

            actions = self.communicator.get_pointed_data_pool_dict(
                agent_name=agent.name,
                data_name=self.action_names,
                start_index=self.start_index,
                end_index=self.end_index
            )

            self.vec_MDP_collector.store_data(
                obs=self.obs,
                action=actions,
                reward=reward,
                next_obs=next_obs,
                mask=mask
            )

            self.obs = next_obs

    def roll(self, agent, rollout_steps):
        self.vec_MDP_collector.reset()
        if self.initial_flag:
            self.initial_flag = False
            if self.sample_enable:
                obs = self.vec_env.reset()

                # push to data pool
                self.communicator.store_data_dict(
                    agent_name=agent.name,
                    data_dict=obs,
                    start_index=self.start_index,
                    end_index=self.end_index
                )

            self.communicator.Barrier()

            if self.optimize_enable:
                self.obs = self.communicator.get_pointed_data_pool_dict(
                    agent_name=agent.name,
                    data_name=self.obs_names,
                    start_index=self.start_index,
                    end_index=self.end_index
                )

        self.communicator.Barrier()

        for _ in range(rollout_steps):
            self.step(agent)
