from AquaML.core.BaseAqua import BaseAqua
from AquaML.core.AgentIOInfo import RLAgentIOInfo
from AquaML.core.Worker import RLAgentWorker, Evaluator, RLVectorEnvWorker
from AquaML.core.RLToolKit import RLVectorEnv, RLBaseEnv, RLNormalVectorEnvWorker
from AquaML.core.ToolKit import SummaryRewardCollector, MDPCollector, mkdir
from AquaML.core.RLToolKit import RLStandardDataSet, Normalization, RewardScaling
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
import os


@dataclass
class LoadFlag:
    actor: bool = False
    critic: bool = False
    state_normalizer: bool = False
    reward_normalizer: bool = False


class AquaRL(BaseAqua):

    def __init__(self,
                 name: str,
                 env,
                 agent,
                 agent_info_dict: dict,
                 decay_lr: bool = False,
                 min_lr: float = 2.5e-5,
                 state_norm: bool = False,
                 reward_norm: bool = False,
                 reset_norm_per=None,
                 snyc_norm_per=1,
                 distributed_norm: bool = False,
                 check_point_path: str = None,
                 load_flag: LoadFlag = LoadFlag(),
                 eval_env=None,
                 comm=None,
                 ):
        """
        单智能体强化学习算法启动器。

        Args:
            name (str): 项目名称。
            env: 环境。两种类型：VecEnv和BaseEnv。
            agent (class): 未实例化的agent类。
            agent_info_dict (dict): agent参数。如使用PPO agent时候, 可以如下设置:
                agent_info_dict = {'actor': actor_model,
                                'critic': critic_model,
                                'agent_params': agent_params,
                                }
                如无特殊需求，agent's name可以使用算法默认值。
            eval_env (BaseEnv, optional): 评估环境。默认为None。在使用VectorEnv时候，eval_env必须存在。
        """

        ########################################
        # AquaRL参数
        ########################################
        self.distributed_norm = distributed_norm

        ########################################
        # 初始化环境参数
        ########################################

        self.decay_lr = decay_lr
        self.min_lr = min_lr

        ########################################
        # 初始化communicator参数
        # 并行方式选择
        ########################################

        if comm is None:
            communicator_info = {
                'type': 'Default',
                'args': {
                    'thread_manager_info': {
                        'type': 'Single',
                        'args': {}
                    }
                }
            }
            self.parrallel_type = 'single'
        else:
            # MPI并行, 主线程配置
            communicator_info = {
                'type': 'Default',
                'args': {
                    'thread_manager_info': {
                        'type': 'MPI',
                        'args': {'comm': comm}
                    }
                }
            }
            self.parrallel_type = 'MPI'

        ########################################
        # 初始化基类
        ########################################
        super().__init__(name=name,
                         communicator_info=communicator_info,
                         )

        self.env = env
        self.reset_norm_per = reset_norm_per

        if eval_env is not None:
            self.eval_env = eval_env
        else:
            self.eval_env = env

        ########################################
        # 初始化agent
        ########################################

        self.agent_params = agent_info_dict['agent_params']

        if 'name' not in agent_info_dict.keys():
            agent_info_dict['name'] = agent.get_algo_name()

        agent_info_dict['level'] = self.level

        self.agent = agent(**agent_info_dict)

        # 获取agent的信息
        agent_name = agent_info_dict['name']
        obs_info = env.obs_info

        # 采样线程获取取
        if self.parrallel_type == 'MPI':
            self.worker_thread = self.communicator.thread_manager.get_total_threads - 1
        else:
            self.worker_thread = 1

        # 获取总的环境数
        self._total_envs = self.env.num_envs * self.worker_thread

        # 计算Roll一次所有线程产生的样本量
        self._total_steps = self.agent_params.rollout_steps * self._total_envs
        self._max_total_steps = self._total_steps * self.agent_params.epochs

        ########################################
        # 初始化env
        ########################################

        # 判断env的类型,及其参数配置
        if isinstance(env, RLVectorEnv):
            self.env_type = 'Vec'
            self.env.set_max_steps(self.agent_params.max_steps)

            self.env_num = self.env.num_envs

            self._communicator_buffer_size = self._total_envs
        elif isinstance(env, RLBaseEnv):
            self.env_type = 'Base'
            self.env_num = 1

            self._communicator_buffer_size = self._total_steps
        else:
            raise TypeError('Env type error!')

        self.agent.env_num = self.env_num

        # 初始化transformer和RNN信息

        actor_out_info = self.agent.actor.output_info
        input_name = self.agent.actor.input_name

        self.env.set_action_state_info(actor_out_info, input_name)

        # 多线程形式下关闭主线程env
        if self.parrallel_type == 'MPI':
            if self.level == 0:
                self.env.close()

        agent_io_info = RLAgentIOInfo(
            agent_name=agent_name,
            obs_info=obs_info.shape_dict,
            obs_type_info=obs_info.type_dict,
            actor_out_info=actor_out_info,
            reward_name=env.reward_info,
            buffer_size=self._communicator_buffer_size,
        )

        self.agent.set_agent_info(agent_io_info)

        # 初始化agent
        self.agent.init()

        ########################################
        # 初始化data pool
        ########################################
        # 初始化之前需要获取indeicate info， parame_info
        self.obs_names = obs_info.shape_dict.keys()

        agent_data_info, param_info, indicate_info = self.agent.get_collection_info(
            reward_info=env.reward_info,
            woker_num=self.worker_thread,
            obs_norm_flag=state_norm,
            reward_norm_flag=reward_norm,
            obs_shape_dict=obs_info.shape_dict,
        )

        # 主线程优先创建
        if self.communicator.thread_manager.get_thread_id > 0:
            import time
            time.sleep(5)

        self.communicator.create_data_collection(
            name=agent_name,
            data_pool_info=agent_data_info,
            param_pool_info=param_info,
            indicate_pool_info=indicate_info,
        )

        # 数据段分配
        if self.parrallel_type == 'single':
            worker_threads = 1
            worker_id = 1
        else:
            worker_threads = self.worker_thread
            worker_id = self.communicator.thread_manager.get_thread_id

        self.communicator.compute_start_end_index(
            data_pool_size=self.communicator.get_data_pool_size(agent_name),
            indicate_pool_size=worker_threads,
            worker_id=worker_id,
            worker_threads=worker_threads,
        )

        ########################################
        # Aqua接口设置
        ########################################
        self._sub_aqua_dict[self.agent.name] = self.agent

        # reward统计接口
        self._reward_names = ('total_reward',)

        ########################################
        # Aqua初始化
        ########################################

        # 文件夹初始化
        self.init_folder()

        # recoder初始化
        self.init_recoder()

        # 初始化线程级别collector后端
        self.mdp_collector = MDPCollector(
            obs_names=agent_io_info.obs_name,
            action_names=self.agent.get_real_policy_out(),
            reward_names=agent_io_info.reward_name,
        )

        self.summary_reward_collector = SummaryRewardCollector(
            reward_names=agent_io_info.reward_name,
        )

        # 分配optimize和sample权限
        if self.parrallel_type == 'single':
            self.optimize_enable = True
            self.sample_enable = True
        elif self.parrallel_type == 'MPI':
            if self.level == 0:
                self.optimize_enable = True
                self.sample_enable = False
            else:
                self.optimize_enable = False
                self.sample_enable = True
            # 创建worker和evaluator

        # TODO:这俩个worker貌似定义不是很合理

        ########################################
        # 创建worker
        ########################################

        if self.env_type == 'Vec':
            if self.distributed_norm:
                self.worker = RLVectorEnvWorker(
                    max_steps=self.agent_params.max_steps,
                    communicator=self.communicator,
                    optimize_enable=self.optimize_enable,
                    sample_enable=self.sample_enable,
                    vec_env=self.env,
                    action_names=self.agent.get_action_names,
                    obs_names=self.env.obs_info.names,
                    reward_names=self.env.reward_info,
                    agent_name=self.agent.name,
                )
            else:
                self.worker = RLNormalVectorEnvWorker(
                    max_steps=self.agent_params.max_steps,
                    communicator=self.communicator,
                    optimize_enable=self.optimize_enable,
                    sample_enable=self.sample_enable,
                    vec_env=self.env,
                    action_names=self.agent.get_action_names,
                    obs_names=self.env.obs_info.names,
                    reward_names=self.env.reward_info,
                    agent_name=self.agent.name,
                )

        else:
            self.worker = RLAgentWorker(
                max_steps=self.agent_params.max_steps,
                env=self.env,
                obs_names=self.env.obs_info.names,
                action_names=self.agent.get_action_names,
                reward_names=self.env.reward_info,
                communicator=self.communicator,
                optimize_enable=self.optimize_enable,
                sample_enable=self.sample_enable,
            )

        self.evaluator = Evaluator()

        # 状态处理插件，暂时仅支持vectorized env

        self._tool_dict['scaler'] = []

        if self.env_type == 'Vec':
            if state_norm:

                self.obs_normalizer = Normalization(
                    obs_shape_dict=self.env.obs_info.shape_dict,
                )

                self.state_norm_flag = True
                self.worker.add_obs_plugin((self.obs_normalizer, self.state_norm_flag))
                self._tool_dict['scaler'].append(self.obs_normalizer)

                if eval_env is not None:
                    self.evaluator.add_obs_plugin(self.obs_normalizer)
            else:
                self.state_norm_flag = False
                self.obs_normalizer = None

            if reward_norm:
                self.reward_normalizer = RewardScaling(1, self.agent_params.gamma)
                self._tool_dict['scaler'].append(self.reward_normalizer)
                self.reward_norm_flag = True
                self.worker.add_reward_plugin((self.reward_normalizer, self.reward_norm_flag))
            else:
                self.reward_normalizer = None
                self.reward_norm_flag = False
            # self.worker.add_obs_plugin(self.obs_normalizer)

        self.snyc_norm_per = snyc_norm_per

        if check_point_path is not None:
            # check the path contains scaler folder if exists, create normalizer and load scaler
            if load_flag.state_normalizer:
                path = os.path.join(check_point_path, 'scaler')
                if self.state_norm_flag is False:
                    self.obs_normalizer = Normalization(
                        obs_shape_dict=self.env.obs_info.shape_dict,
                    )

                    self.worker.add_obs_plugin((self.obs_normalizer, self.state_norm_flag))

                    if eval_env is not None:
                        self.evaluator.add_obs_plugin(self.obs_normalizer)

                self.obs_normalizer.load(path)

            if load_flag.reward_normalizer:
                path = os.path.join(check_point_path, 'scaler')
                if self.reward_norm_flag is False:
                    self.reward_normalizer = RewardScaling(1, self.agent_params.gamma)
                    self.worker.add_reward_plugin((self.reward_normalizer, self.reward_norm_flag))

                self.reward_normalizer.load(path)

            if load_flag.actor:
                path = os.path.join(check_point_path, 'actor.h5')
                self.agent.actor.load_weights(path)

            if load_flag.critic:
                if self.optimize_enable:
                    path = os.path.join(check_point_path, 'critic.h5')
                    self.agent.critic.load_weights(path)

    def sampling(self):

        std_data_set = RLStandardDataSet(
            rollout_steps=self.agent_params.rollout_steps,
            num_envs=self._total_envs,
        )

        self.worker.roll(
            agent=self.agent,
            rollout_steps=self.agent_params.rollout_steps,
            std_data_set=std_data_set,
        )

        return std_data_set

    def evaluate(self):
        """
        评估。
        """
        self.evaluator.roll(
            env=self.eval_env,
            agent=self.agent,
            episode_length=self.agent_params.eval_episode_length,
            episodes=self.agent_params.eval_episodes,
            collector=self.summary_reward_collector,
        )

        ########################################
        # 将数据推送至data pool
        ########################################

        # 获取数据，这里可以加入数据处理backend

        reward_dict = self.summary_reward_collector.get_data()

        # 获取起始index
        star_index = self.communicator.indicate_pool_end_index

        self.communicator.store_indicate_dict(
            agent_name=self.agent.name,
            indicate_dict=reward_dict,
            index=star_index,
            # pre_fix='summary_',
        )

    def sync(self):
        """
        同步。
        """
        self.sync_param_pool()
        self.sync_model_tool()

    def run(self):
        """
        运行。
        """
        # TODO: 接口不完善需要统一
        # if self.optimize_enable:
        # if self.env_type != 'Vec':
        #     self.sync()
        # self.sync()

        if self.optimize_enable:
            self.sync()
        runner_mode = 'on-policy'
        for epoch in range(self.agent_params.epochs):

            self.communicator.thread_manager.Barrier()

            if self.sample_enable:
                self.sync()

                # sync norm
                if self.env_type == 'Vec':
                    if (epoch + 1) % self.snyc_norm_per == 0 and self.distributed_norm:
                        indicate_data = self.communicator.get_indicate_pool_dict(self.agent.name)
                        obs_norm_dict = {}

                        if self.state_norm_flag:
                            for name in self.obs_names:
                                mean = indicate_data[name + '_mean'][0]
                                std = indicate_data[name + '_std'][0]

                                obs_norm_dict[name] = (mean, std)

                                self.obs_normalizer.set_data(obs_norm_dict)

                        if self.reward_norm_flag:
                            total_reward_mean = indicate_data['total_reward_mean'][0]
                            total_reward_std = indicate_data['total_reward_std'][0]

                            self.reward_normalizer.set_data((total_reward_mean, total_reward_std))

            self.communicator.thread_manager.Barrier()

            std_data_set = self.sampling()

            # print("{}: {}".format(self.communicator.get_level(), self.agent.log_std.buffer))

            if self.sample_enable:

                # 获取子线程normalize的数据
                if self.env_type == 'Vec':
                    if self.state_norm_flag:
                        self.obs_normalizer.push_to_communicator(
                            communicator=self.communicator,
                            agent_name=self.agent.name,
                        )
                    if self.reward_norm_flag:
                        self.reward_normalizer.push_to_communicator(
                            communicator=self.communicator,
                            agent_name=self.agent.name,
                        )

            self.communicator.thread_manager.Barrier()

            if self.optimize_enable:
                print('####################{}####################'.format(epoch + 1))
                loss_tracker, reward_info = self.agent.optimize(std_data_set, runner_mode)

                loss_info = loss_tracker.get_data()
                del std_data_set

                for key, value in loss_info.items():
                    print(key, value)

                for key, value in reward_info.items():
                    print(key, value)

                current_steps = self.agent_params.rollout_steps * self._total_envs * (epoch + 1)

                if self.decay_lr:
                    for name, param in self.agent.get_optimizer_pool.items():
                        optimizer = param['optimizer']
                        lr = param['lr']
                        lr_now = lr * (1 - current_steps / self._max_total_steps)
                        lr_now = max(lr_now, self.min_lr)
                        optimizer.learning_rate.assign(lr_now)
                        print('{} lr:{}'.format(name, lr_now))

                self.sync()

                self.recoder.record_scalar(reward_info, epoch + 1)
                self.recoder.record_scalar(loss_info, epoch + 1)

                if self.env_type == 'Vec':

                    indicate_data = self.communicator.get_indicate_pool_dict(self.agent.name)
                    obs_norm_dict = {}

                    if self.state_norm_flag and self.distributed_norm:
                        for name in self.obs_names:
                            mean = np.mean(indicate_data[name + '_mean'], axis=0)

                            std_2 = np.sum(indicate_data[name + '_std'] ** 2, axis=0)
                            std = std_2 / self.worker_thread
                            std = np.sqrt(std)

                            obs_norm_dict[name] = (mean, std)

                            self.communicator.store_indicate_dict(
                                agent_name=self.agent.name,
                                indicate_dict={name + '_mean': mean, name + '_std': std},
                                index=0,
                            )

                            self.obs_normalizer.set_data(obs_norm_dict)

                    # total_reward = np.mean(indicate_data['total_reward'])

                    if self.reward_norm_flag and self.distributed_norm:
                        total_reward_mean = np.mean(indicate_data['total_reward_mean'])
                        total_reward_std = np.mean(indicate_data['total_reward_std'])

                        self.reward_normalizer.set_data((total_reward_mean, total_reward_std))

                        self.communicator.store_indicate_dict(
                            agent_name=self.agent.name,
                            indicate_dict={'total_reward_mean': total_reward_mean,
                                           'total_reward_std': total_reward_std},
                            index=0,
                        )

                    if self.reset_norm_per is not None:
                        if (epoch + 1) % self.reset_norm_per == 0:
                            # self.obs_normalizer.reset()
                            if self.reward_norm_flag:
                                self.reward_normalizer.reset()

                    for key, value in self._tool_dict.items():
                        cache_path = os.path.join(self.file_system.get_cache_path(self.agent.name), key)
                        # th_id = self.communicator.thread_manager.get_thread_id
                        # cache_path = os.path.join(cache_path, str(th_id))
                        for tool in value:
                            tool.save(cache_path)

                if epoch % self.agent_params.checkpoint_interval == 0:
                    self.recoder.save_checkpoint(
                        model_dict=self.agent.get_all_model_dict,
                        epoch=epoch + 1,
                        checkpoint_dir=self.file_system.get_history_model_path(self.agent.name),
                        tool=self._tool_dict,
                    )

                # sync normalizer

            for key, value in self._tool_dict.items():
                cache_path = os.path.join(self.file_system.get_cache_path(self.agent.name), key)
                th_id = self.communicator.thread_manager.get_thread_id
                cache_path = os.path.join(cache_path, str(th_id))
                for tool in value:
                    tool.save(cache_path)

            if epoch % self.agent_params.checkpoint_interval == 0:
                history = self.file_system.get_history_model_path(self.agent.name)
                history_path = os.path.join(history, str(epoch + 1))
                for key, value in self._tool_dict.items():
                    cache_path = os.path.join(history_path, key)
                    th_id = self.communicator.thread_manager.get_thread_id
                    cache_path = os.path.join(cache_path, str(th_id))
                    for tool in value:
                        tool.save(cache_path)

            self.communicator.thread_manager.Barrier()

            if (epoch + 1) % self.agent_params.eval_interval == 0:

                if self.sample_enable:
                    self.sync()
                    self.evaluate()

                self.communicator.thread_manager.Barrier()

                if self.optimize_enable:
                    # 汇总数据
                    summery_dict = self.communicator.get_indicate_pool_dict(self.agent.name)

                    # 计算平均值
                    new_summery_dict = {}
                    pre_fix = 'reward/'
                    for key, value in summery_dict.items():
                        # if 'reward' in key:
                        if 'max' in key:
                            new_summery_dict[pre_fix + key] = np.max(value)
                        elif 'min' in key:
                            new_summery_dict[pre_fix + key] = np.min(value)
                        else:
                            new_summery_dict[pre_fix + key] = np.mean(value)

                    # 记录数据
                    for key, value in new_summery_dict.items():
                        print(key, value)

                    self.recoder.record_scalar(new_summery_dict, epoch + 1)

    def run_off_policy(self):
        """
        运行。
        """
        # TODO: 接口不完善需要统一
        # if self.optimize_enable:
        # if self.env_type != 'Vec':
        #     self.sync()
        # self.sync()

        if self.optimize_enable:
            self.sync()

        runner_mod = 'off-policy'

        for epoch in range(self.agent_params.epochs + self.agent_params.learning_starts):
            std_data_set = self.sampling()

            loss_tracker, _ = self.agent.optimize(std_data_set, runner_mod)

            # 每十次保存最新模型
            if epoch % 10 == 0:
                self.sync()

            if epoch % self.agent_params.checkpoint_interval == 0 and self.agent.eval_flag:
                self.recoder.save_checkpoint(
                    model_dict=self.agent.get_all_model_dict,
                    epoch=epoch + 1,
                    checkpoint_dir=self.file_system.get_history_model_path(self.agent.name),
                    tool=self._tool_dict,
                )

            if (epoch + 1) % self.agent_params.eval_interval == 0 and self.agent.eval_flag:

                if self.sample_enable:
                    self.sync()
                    self.evaluate()

                self.communicator.thread_manager.Barrier()

                if self.optimize_enable:

                    loss_data = loss_tracker.get_data()
                    # 汇总数据
                    summery_dict = self.communicator.get_indicate_pool_dict(self.agent.name)

                    print('####################{}####################'.format(epoch + 1))

                    # 计算平均值
                    new_summery_dict = {}
                    pre_fix = 'reward/'
                    for key, value in summery_dict.items():
                        # if 'reward' in key:
                        if 'max' in key:
                            new_summery_dict[pre_fix + key] = np.max(value)
                        elif 'min' in key:
                            new_summery_dict[pre_fix + key] = np.min(value)
                        else:
                            new_summery_dict[pre_fix + key] = np.mean(value)

                    pre_fix = 'loss/'
                    for key, value in loss_data.items():
                        # if 'reward' in key:
                        if 'max' in key:
                            new_summery_dict[pre_fix + key] = np.max(value)
                        elif 'min' in key:
                            new_summery_dict[pre_fix + key] = np.min(value)
                        else:
                            new_summery_dict[pre_fix + key] = np.mean(value)

                    # 记录数据
                    for key, value in new_summery_dict.items():
                        print(key, value)

                    self.recoder.record_scalar(new_summery_dict, epoch + 1)

    def run_offline(self, display_times=10):
        """
        运行。
        """
        # TODO: 接口不完善需要统一
        # if self.optimize_enable:
        # if self.env_type != 'Vec':
        #     self.sync()
        # self.sync()

        for epoch in range(self.agent_params.epochs):
            # print("{}: {}".format(self.communicator.get_level(), self.agent.log_std.buffer))
            loss_tracker, _ = self.agent.optimize()
            self.sync()

            if (epoch + 1) % display_times == 0:
                summery_dict = loss_tracker.get_data()
                print('####################{}####################'.format(epoch + 1))

                for key, value in summery_dict.items():
                    print(key, value)

                self.recoder.record_scalar(summery_dict, epoch + 1)

                # 存储最新模型

                if epoch % self.agent_params.checkpoint_interval == 0:
                    self.recoder.save_checkpoint(
                        model_dict=self.agent.get_all_model_dict,
                        epoch=epoch + 1,
                        checkpoint_dir=self.file_system.get_history_model_path(self.agent.name),
                        tool=self._tool_dict,
                    )
