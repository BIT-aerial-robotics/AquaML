"""
V2版本将逐渐替代V1版本，V2版本将使用更加简洁的代码，更加易于理解。

将Worker的参数从类中分离出来，使得Worker可以更加灵活的使用。也方变未来用于课程学习的操作。
"""


class RLWorker:
    def __init__(self, env, max_steps, update_interval, summary_episodes):

        # 环境基本参数
        self.obs = None
        self.env = env
        self.reset_flag = True
        self.episode_step_count = 0

        # roll out 参数
        self.update_interval = update_interval
        self.max_steps = max_steps

        # summary 参数
        self.summary_episodes = summary_episodes

    def step(self, algo, test_flag=False):
        """
        roll out the environment and get data.
        when step_count == update_interval, need to update the model

        :param algo: algo提供网络动作执行方式
        """
        # reset the environment and actor model
        # when first step or reset flag is True
        if self.reset_flag:
            self.obs, flag = self.env.reset()
            self.reset_flag = False
            if flag:
                algo.actor.reset()  # 重置actor
            self.episode_step_count = 0

        action_dict = algo.get_action(self.obs, test_flag=test_flag)

        obs_, reward, done, info = self.env.step(action_dict)

        self.episode_step_count += 1

        if self.episode_step_count >= self.max_steps:
            done = True

        if done:
            self.reset_flag = True
            mask = 0
        else:
            mask = 1

        # if not test_flag:
        algo.store_data(obs=self.obs, action=action_dict,
                            reward=reward, next_obs=obs_, mask=mask)

        self.obs = obs_

    def roll(self, algo, test_flag=False):
        """
        roll out the environment and get data.
        when step_count == update_interval, need to update the model
        """

        for _ in range(self.update_interval):
            self.step(algo, test_flag=test_flag)