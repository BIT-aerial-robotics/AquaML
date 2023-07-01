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
            next_obs, reward, done, computing_obs = self.vec_env.step(actions)

            self.communicator.store_data_dict(
                agent_name=agent.name,
                data_dict=computing_obs,
                start_index=self.start_index,
                end_index=self.end_index
            )

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
                data_name=self.next_obs_names,
                start_index=self.start_index,
                end_index=self.end_index
            )

            computing_obs = self.communicator.get_pointed_data_pool_dict(
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
                obs=deepcopy(self.obs),
                action=deepcopy(actions),
                reward=deepcopy(reward),
                next_obs=deepcopy(next_obs),
                mask=deepcopy(mask['mask'])
            )

            self.obs = deepcopy(computing_obs)

    def roll(self, agent, rollout_steps, std_data_set: RLStandardDataSet):
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
