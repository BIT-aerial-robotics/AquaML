from abc import ABC, abstractmethod
from AquaML.core.RLToolKit import VecCollector, VecMDPCollector, RLStandardDataSet
from AquaML.core.ToolKit import SummaryRewardCollector, MDPCollector
import numpy as np
from copy import deepcopy, copy
import tensorflow as tf


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
                 env,
                 obs_names,
                 action_names,
                 reward_names,
                 communicator,
                 sample_enable,
                 optimize_enable,
                 ):
        self.env = env
        # 参数设置
        self.max_steps = max_steps

        self.communicator = communicator

        self.obs_names = obs_names
        self.action_names = action_names
        self.reward_names = reward_names

        # mdp collector
        self.collector = MDPCollector(
            obs_names=obs_names,
            action_names=action_names,
            reward_names=reward_names,
        )

        self.sample_enable = sample_enable
        self.optimize_enable = optimize_enable

        # 运行过程参数
        self.reset_flag = True
        self.obs = None
        self.episode_step_count = 0

        self.obs = None

    def step(self, env, agent, step, rollout_steps):
        """
        采样一次数据。
        """

        action_dict = agent.get_action(self.obs)

        obs_, reward, done, info = env.step(action_dict)

        self.episode_step_count += 1

        if self.episode_step_count >= self.max_steps:
            done = True

        if done:
            mask = 0
        else:
            mask = 1

        if done:
            computing_obs, flag = env.reset()

            # TODO: aditional reset 这个地方作为未来的接口
            self.episode_step_count = 0
        else:
            computing_obs = obs_

        self.collector.store_data(
            obs=self.obs,
            action=action_dict,
            reward=reward,
            next_obs=obs_,
            mask=mask
        )

        self.obs = computing_obs

    def roll(self, agent, rollout_steps, std_data_set: RLStandardDataSet):
        if self.reset_flag:
            computing_obs, flag = self.env.reset()
            self.reset_flag = False
            self.episode_step_count = 0
            self.obs = computing_obs

        if self.sample_enable:
            self.collector.reset()

            for step in range(rollout_steps):
                self.step(
                    env=self.env,
                    agent=agent,
                    step=step,
                    rollout_steps=rollout_steps
                )

            obs_dict, action_dict, reward_dict, next_obs_dict, mask = self.collector.get_data()

            start_index = self.communicator.data_pool_start_index
            end_index = start_index + rollout_steps

            # 去除action中hidden state
            new_action_dict = {}
            for key, value in action_dict.items():
                if 'hidden_' not in key:
                    new_action_dict[key] = value

            # 推送数据
            self.communicator.store_data_dict(
                agent_name=agent.name,
                data_dict=obs_dict,
                start_index=start_index,
                end_index=end_index,
            )

            self.communicator.store_data_dict(
                agent_name=agent.name,
                data_dict=action_dict,
                start_index=start_index,
                end_index=end_index,
            )

            self.communicator.store_data_dict(
                agent_name=agent.name,
                data_dict=reward_dict,
                start_index=start_index,
                end_index=end_index,
            )

            self.communicator.store_data_dict(
                agent_name=agent.name,
                data_dict=next_obs_dict,
                start_index=start_index,
                end_index=end_index,
            )

            self.communicator.store_data_dict(
                agent_name=agent.name,
                data_dict={'mask': mask},
                start_index=start_index,
                end_index=end_index,
            )

        self.communicator.thread_manager.Barrier()
        if self.optimize_enable:
            # 获取所有数据并且按照规定格式整理
            num_envs = std_data_set.num_envs

            buffer = self.communicator.get_data_pool_dict(agent.name)

            obs_dict = {}
            action_dict = {}
            reward_dict = {}
            next_obs_dict = {}

            for key in self.obs_names:
                obs_dict[key] = np.reshape(buffer[key], (num_envs, rollout_steps, -1))
                next_obs_dict['next_' + key] = np.reshape(buffer['next_' + key], (num_envs, rollout_steps, -1))

            for key in self.action_names:
                action_dict[key] = np.reshape(buffer[key], (num_envs, rollout_steps, -1))

            for key in self.reward_names:
                reward_dict[key] = np.reshape(buffer[key], (num_envs, rollout_steps, -1))

            mask = np.reshape(buffer['mask'], (num_envs, rollout_steps, -1))

            std_data_set(
                obs=obs_dict,
                action=action_dict,
                reward=reward_dict,
                next_obs=next_obs_dict,
                mask=mask,
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

        # 插件接口
        self._obs_plugin = []

    def add_obs_plugin(self, obs_plugin):
        self._obs_plugin.append(obs_plugin)

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
                 agent_name: str,
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

        self.next_obs_names = ['next_' + name for name in obs_names]

        if self.thread_level == 0:
            # main process
            self.start_index = 0
            self.end_index = self.communicator.get_data_pool_size(agent_name)

        else:
            # sub process
            self.start_index = self.communicator.data_pool_start_index
            self.end_index = self.communicator.data_pool_end_index

        self.initial_flag = True

        self.vec_MDP_collector = VecMDPCollector(
            obs_names=self.obs_names,
            reward_names=self.reward_names,
            action_names=self.action_names,
            next_obs_names=self.next_obs_names,
        )

        # 插件接口
        self._obs_plugin = []
        self._reward_plugin = []

    def add_obs_plugin(self, obs_plugin):
        self._obs_plugin.append(obs_plugin)

    def add_reward_plugin(self, reward_plugin):
        self._reward_plugin.append(reward_plugin)

    def step(self, agent):

        if self.optimize_enable:
            # format of data: (mxn, ...)
            # m: number of threads, n: number envs in each thread

            actions = agent.get_action(self.obs)

            # print(actions['action'][0])

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
            actions_ = self.communicator.get_pointed_data_pool_dict(
                agent_name=agent.name,
                data_name=self.action_names,
                start_index=self.start_index,
                end_index=self.end_index
            )

            # print(actions['action'][0])

            # step
            next_obs, reward, done, computing_obs = self.vec_env.step(deepcopy(actions_))

            for obs_plugin, args in self._obs_plugin:
                next_obs_ = obs_plugin(next_obs, args)
                next_obs.update(next_obs_)
                computing_obs_ = obs_plugin(computing_obs, False)
                computing_obs.update(computing_obs_)

            new_next_obs = {}
            for key in next_obs.keys():
                new_next_obs['next_' + key] = next_obs[key]

            for reward_plugin, args in self._reward_plugin:
                reward_ = reward_plugin(reward['total_reward'])
                # if 'indicate' not in reward.keys():
                reward['total_reward'] = deepcopy(reward_)

            self.communicator.store_data_dict(
                agent_name=agent.name,
                data_dict=computing_obs,
                start_index=self.start_index,
                end_index=self.end_index
            )

            self.communicator.store_data_dict(
                agent_name=agent.name,
                data_dict=new_next_obs,
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
            next_obs_ = self.communicator.get_pointed_data_pool_dict(
                agent_name=agent.name,
                data_name=self.next_obs_names,
                start_index=self.start_index,
                end_index=self.end_index
            )

            computing_obs_ = self.communicator.get_pointed_data_pool_dict(
                agent_name=agent.name,
                data_name=self.obs_names,
                start_index=self.start_index,
                end_index=self.end_index
            )

            reward_ = self.communicator.get_pointed_data_pool_dict(
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

            actions_ = self.communicator.get_pointed_data_pool_dict(
                agent_name=agent.name,
                data_name=self.action_names,
                start_index=self.start_index,
                end_index=self.end_index
            )

            self.vec_MDP_collector.store_data(
                obs=deepcopy(self.obs),
                action=deepcopy(actions_),
                reward=deepcopy(reward_),
                next_obs=deepcopy(next_obs_),
                mask=deepcopy(mask['mask'])
            )

            self.obs = deepcopy(computing_obs_)

    def roll(self, agent, rollout_steps, std_data_set: RLStandardDataSet):
        self.vec_MDP_collector.reset()

        # if self.sample_enable:
        #     obs = self.vec_env.reset()
        #
        #     for obs_plugin, args in self._obs_plugin:
        #         obs_ = obs_plugin(obs, args)
        #         obs.update(obs_)
        #
        #     # push to data pool
        #     self.communicator.store_data_dict(
        #         agent_name=agent.name,
        #         data_dict=obs,
        #         start_index=self.start_index,
        #         end_index=self.end_index
        #     )
        #
        #     self.communicator.Barrier()
        #
        # if self.optimize_enable:
        #     obs = self.communicator.get_pointed_data_pool_dict(
        #         agent_name=agent.name,
        #         data_name=self.obs_names,
        #         start_index=self.start_index,
        #         end_index=self.end_index
        #     )
        #
        #     self.obs = deepcopy(obs)
        #
        # self.communicator.Barrier()

        if self.initial_flag:
            self.initial_flag = False
            if self.sample_enable:
                obs = self.vec_env.reset()

                for obs_plugin, args in self._obs_plugin:
                    obs_ = obs_plugin(obs, args)
                    obs.update(obs_)

                # push to data pool
                self.communicator.store_data_dict(
                    agent_name=agent.name,
                    data_dict=obs,
                    start_index=self.start_index,
                    end_index=self.end_index
                )

            self.communicator.Barrier()

            if self.optimize_enable:
                obs = self.communicator.get_pointed_data_pool_dict(
                    agent_name=agent.name,
                    data_name=self.obs_names,
                    start_index=self.start_index,
                    end_index=self.end_index
                )

                self.obs = deepcopy(obs)

        self.communicator.Barrier()

        for _ in range(rollout_steps):
            self.step(agent)

        if self.optimize_enable:
            obs_dict, action_dict, reward_dict, next_obs_dict, mask = self.vec_MDP_collector.get_data()

            std_data_set(
                obs=obs_dict,
                action=action_dict,
                reward=reward_dict,
                next_obs=next_obs_dict,
                mask=mask
            )