from abc import ABC, abstractmethod


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

    def step(self, env, agent, collector):
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

        for _ in range(rollout_steps):
            self.step(
                env=env,
                agent=agent,
                collector=collector
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
