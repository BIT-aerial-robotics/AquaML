class RLWorker:
    """ RLWorker is used to interact with environment and get data.

    It can be used in on-policy and off-policy reinforcement learning.

    """

    def __init__(self, rl_algo):
        self.summary_reward_dict = None
        self.rl_algo = rl_algo
        self.reset_flag = True

        # get information from rl_algo
        self.env = rl_algo.env

        self.update_interval = rl_algo.update_interval

        self.initial_summary_reward()

        self.record_summary_times = 0

        self.summary_store_bias = self.rl_algo.sample_id * self.rl_algo.each_thread_summary_episodes

        self.max_steps = self.rl_algo.hyper_parameters.epoch_length

        self.obs = None
        self.step_count = 0
        self.episode_step_count = 0

    # TODO: 加入最大步数限制
    def step(self, test_flag=False):

        # reset the environment and actor model
        # when first step or reset flag is True
        if self.reset_flag:
            self.obs, flag = self.env.reset()
            self.reset_flag = False
            if flag:
                self.rl_algo.actor.reset()
            # self.rl_algo.actor.reset()
            self.episode_step_count = 0
            self.initial_summary_reward()

        action_dict = self.rl_algo.get_action(self.obs, test_flag=test_flag)

        obs_, reward, done, info = self.env.step(action_dict)  # obs, reward are dict

        # print(action_dict['action'])

        self.record_summary_reward(reward)

        self.step_count += 1

        self.episode_step_count += 1

        if self.episode_step_count >= self.max_steps:
            done = True

        # done flag is True, need to reset the environment
        if done:
            self.reset_flag = True
            mask = 0

            # store reward in summary_* data pool
            index = self.record_summary_times % self.rl_algo.each_thread_summary_episodes + self.summary_store_bias
            self.rl_algo.data_pool.store(self.summary_reward_dict, index=index)
            self.record_summary_times += 1

        else:
            mask = 1

        # store the data
        # TODO: remove this line
        # reward['total_reward'] = (reward['total_reward'] + 8) / 8
        if not test_flag:
            self.rl_algo.store_data(obs=self.obs, action=action_dict,
                                    reward=reward, next_obs=obs_, mask=mask)
            self.step_count = self.rl_algo.store_counter

        self.obs = obs_

    def roll(self, update_interval, test_flag=False):
        """roll the environment and get data.
        when step_count == update_interval, need to update the model
        """

        for _ in range(update_interval):
            self.step(test_flag=test_flag)

        if test_flag:
            if update_interval>0:
                self.env.display()

    def summary_reward(self):
        pass

    def initial_summary_reward(self):
        self.summary_reward_dict = {}
        for key in self.env.reward_info:
            self.summary_reward_dict['summary_' + key] = 0

    def record_summary_reward(self, reward_dict):
        for key in reward_dict:
            self.summary_reward_dict['summary_' + key] += reward_dict[key]
