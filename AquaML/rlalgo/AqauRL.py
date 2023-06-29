from AquaML.core.BaseAqua import BaseAqua
from AquaML.core.AgentIOInfo import RLAgentIOInfo
from AquaML.core.Worker import RLAgentWorker, Evaluator, RLVectorEnvWorker
from AquaML.core.RLToolKit import RLVectorEnv, RLBaseEnv
from AquaML.core.ToolKit import SummaryRewardCollector, MDPCollector
from AquaML.core.RLToolKit import RLStandardDataSet
import numpy as np


class AquaRL(BaseAqua):

    def __init__(self,
                 name: str,
                 env,
                 agent,
                 agent_info_dict: dict,
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

        if eval_env is not None:
            self.eval_env = eval_env
        else:
            self.eval_env = env

        # 判断env的类型
        if isinstance(env, RLVectorEnv):
            self.env_type = 'Vec'
        elif isinstance(env, RLBaseEnv):
            self.env_type = 'Base'
        else:
            raise TypeError('Env type error!')

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

        ########################################
        # 初始化env
        ########################################

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
            buffer_size=self._total_steps,
        )

        self.agent.set_agent_info(agent_io_info)

        # 初始化agent
        self.agent.init()

        ########################################
        # 初始化data pool
        ########################################
        # 初始化之前需要获取indeicate info， parame_info

        agent_data_info, param_info, indicate_info = self.agent.get_collection_info(
            reward_info=env.reward_info,
            woker_num=self.worker_thread,
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
        if self.env_type == 'Vec':
            self.worker = RLVectorEnvWorker(
                max_steps=self.agent_params.max_steps,
                communicator=self.communicator,
                optimize_enable=self.optimize_enable,
                sample_enable=self.sample_enable,
                vec_env=self.env,
                action_names=self.agent.get_action_names,
                obs_names=self.env.obs_info.keys(),
                reward_names=self.env.reward_info,

            )
        else:
            self.worker = RLAgentWorker(
                max_steps=self.agent_params.max_steps,
                env=self.env,
                obs_names=self.env.obs_info.keys(),
                action_names=self.agent.get_action_names,
                reward_names=self.env.reward_info,
                communicator=self.communicator,
                optimize_enable=self.optimize_enable,
                sample_enable=self.sample_enable,
            )

        self.evaluator = Evaluator()

    def sampling(self):
        pass

    def _obtain_data(self):
        """
        获取数据。
        """
        self.worker.roll(
            agent=self.agent,
            rollout_steps=self.agent_params.rollout_steps,
            collector=self.mdp_collector,
        )

        ########################################
        # 将数据推送至data pool
        ########################################

        # 获取数据，这里可以加入数据处理backend

        obs_dict, action_dict, reward_dict, next_obs_dict, mask = self.mdp_collector.get_data()

        # 去除action中hidden state
        new_action_dict = {}
        for key, value in action_dict.items():
            if 'hidden_' not in key:
                new_action_dict[key] = value

        # 获取起始index
        star_index = self.communicator.data_pool_start_index

        length = action_dict['action'].shape[0]

        end_index = star_index + length

        # 推送数据
        self.communicator.store_data_dict(
            agent_name=self.agent.name,
            data_dict=obs_dict,
            start_index=star_index,
            end_index=end_index,
        )

        self.communicator.store_data_dict(
            agent_name=self.agent.name,
            data_dict=new_action_dict,
            start_index=star_index,
            end_index=end_index,
        )

        self.communicator.store_data_dict(
            agent_name=self.agent.name,
            data_dict=reward_dict,
            start_index=star_index,
            end_index=end_index,
        )

        self.communicator.store_data_dict(
            agent_name=self.agent.name,
            data_dict=next_obs_dict,
            start_index=star_index,
            end_index=end_index,
        )

        self.communicator.store_data_dict(
            agent_name=self.agent.name,
            data_dict={'mask': mask},
            start_index=star_index,
            end_index=end_index,
        )

    def _obtain_env_vec(self):
        """
        获取数据。
        """
        self.worker.roll(
            agent=self.agent,
            rollout_steps=self.agent_params.rollout_steps,
        )

        # 获取数据，这里可以加入数据处理backend,主线程
        if self.optimize_enable:
            data = self.worker.get_data()
    

    def evaluate(self):
        """
        评估。
        """
        self.evaluator.roll(
            env=self.env,
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
        if self.optimize_enable:
            self.sync()

        for epoch in range(self.agent_params.epochs):

            self.communicator.thread_manager.Barrier()

            if self.sample_enable:
                self.sync()
                self.obtain_data()

            self.communicator.thread_manager.Barrier()

            if self.optimize_enable:
                print('####################{}####################'.format(epoch + 1))
                loss_info = self.agent.optimize(self.communicator)

                for key, value in loss_info.items():
                    print(key, value)

                self.sync()

            self.communicator.thread_manager.Barrier()

            # 评估主线程backbone,用于统计输出，记录
            if self.optimize_enable:
                # 获取当前epoch reward
                episode_summery_dict = {}
                ep_length = []

                data_dict = self.communicator.get_data_pool_dict(self.agent.name)

                # 获取当前epoch reward
                epoch_reward_dict = dict()

                for name in self._reward_names:
                    epoch_reward_dict[name] = data_dict[name]

                masks = data_dict['mask']

                index_done = np.where(masks == 0)[0] + 1

                start_index = 0

                pre_fix = 'ep_reward/'
                for end_index in index_done:
                    for name in self._reward_names:
                        if pre_fix + name not in episode_summery_dict:
                            episode_summery_dict[pre_fix + name] = []
                        episode_summery_dict[pre_fix + name].append(
                            np.sum(epoch_reward_dict[name][start_index:end_index]))

                    ep_length.append(end_index - start_index)

                    start_index = end_index

                episode_summery_dict[pre_fix + 'ep_length'] = ep_length
                # 计算平均值
                new_episode_summery_dict = {}

                for key, value in episode_summery_dict.items():
                    new_episode_summery_dict[key] = np.mean(value)
                    new_episode_summery_dict[key + '_max'] = np.max(value)
                    new_episode_summery_dict[key + '_min'] = np.min(value)

                new_episode_summery_dict[pre_fix + 'ep_num'] = len(ep_length)

                for key, value in new_episode_summery_dict.items():
                    print(key, value)

                self.recoder.record_scalar(new_episode_summery_dict, epoch + 1)

                self.recoder.record_scalar(loss_info, epoch + 1)

                self.recoder.record_model_weight(self.agent.get_all_model_dict, epoch + 1)

                # TODO: 统一checkpoint接口（history, best, latest）

                if (epoch + 1) % self.agent_params.checkpoint_interval == 0:
                    self.recoder.save_checkpoint(
                        model_dict=self.agent.get_all_model_dict,
                        epoch=epoch + 1,
                        checkpoint_dir=self.file_system.get_history_model_path(self.agent.name),
                    )

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
