from AquaML.param.DataInfo import DataInfo
from AquaML import DataModule
from AquaML.communicator.CommunicatorBase import CommunicatorBase
from AquaML.framework.RealUpdaterStarter import RealUpdaterStarter
from AquaML.core.old.FileSystem import DefaultFileSystem
import numpy as np

'''
# TODO: 将部分功能迁入Base中。
'''


class EvolutionML:
    """
    1. 交互模块定义，交互模块用于和真机交互的策略。可以是网络策略，也可以是PID控制策略。

        交互策略定义形式如下：
        {
            'policy_name': {
                'policy': policy class,
                'param': policy param,
                'updatable': True/False,
                'sub_process_id': id,
                'GPU_enable': False,
            }
            
        }
        定义模式将会逐渐完善。
        
        每个策略默认按照在字典中的顺序进行编号。
        
        为了更方便的配置真机策略对计算资源的控制,我们使用sub_process_id来控制，当sub_process_id相同时，表示几个策略共享一个进程。
    """

    def __init__(self, task_name: str,
                 env_info: DataInfo,
                 real_policies: dict,
                 policy_updater: dict,
                 robot_api: dict,
                 policy_selector: dict,
                 capacity: int,
                 communicator: CommunicatorBase,
                 process_config_path: str,
                 offline_dataset_path: str = None,
                 ):
        """
        用于初始化真机在线学习模块。
        
        real policies用于和真机交互的一组策略，然后使用特定的候选策略选出最合适的。real policies输出
        组动作。候选策略模块选出最合适的动作。
            
        real_policies的定义形式如下：
        {
            'policy_name': {
                'policy': policy class,
                'param': policy param,
                
            }
        }
        
        policy_updater的定义形式如下：
        {
            'policy_name': {
                'updater': policy updater class,
                'param': policy updater param,
        }
        }
        
        robot_api的定义形式如下：
        {
            'api_name': {
                'api': api class,
                'param': api param,
        }
        }
        
        policy_selector的定义形式如下：
        {
            'selector_name': {
                'selector': selector class,
                'param': selector param,
        }
        
        }
         
        
        real policies -> id action-> actions -> candidate action policy -> action

        注意：update policy与其对应的策略名称相同。在policy_updater中，policy updater的名称需要与参数中的名称相同。

        Args:
            task_name(str): 任务名称
            env_info (DataInfo): 机器人环境信息,另外的功能就是具有数据信息的功能。
            real_policies (dict): 真实策略。
            policy_updater (dict): 策略更新器。当前仅支持一个updater.
            robot_api (dict): 与机器人交互的API。
            policy_selector (dict): 策略选择器。
            communicator (CommunicatorBase): 通信模块。用于多进程通信，配置多进程通信模块。
            process_config_path (str): 进程配置文件路径。用于配置进程的一些参数。
        """

        ############################## 共有属性 ##############################
        self._task_name = task_name
        self._communicator = communicator
        self._env_info = env_info
        self._capacity = capacity
        self._offline_dataset_path = offline_dataset_path

        self._current_state_name_list = []  # 当前状态group
        self._current_state_name_map_list = []  # 当前状态group
        self._next_state_name_list = []  # 下一个状态group
        self._next_state_name_map_list = []  # 下一个状态group
        self._reward_name_list = []
        self._action_name_list = []

        self._candidate_action_name_list = []  # 候选动作group
        self._candidate_action_name = None
        # self._control_action_name_list = [] # 控制动作group
        self._control_action_name = None

        self._scope_names_list = []  # scope名称列表
        self._scope_names_map_list = []  # scope名称映射列表
        self._data_names_in_buffer_list = []  # buffer中的数据名称列表

        # env_info rl的初始化
        rl_flag_dict = self._env_info.rl_init(policy_num=len(real_policies))

        ############################# 参数检查 #############################

        # 检查policy_updater的名称是否和参数中的名称相同
        for policy_updater_name, policy_updater_info in policy_updater.items():

            param = policy_updater_info['param']
            if 'name' not in param:
                self._communicator.logger_warning(
                    'EvolutionML ' + 'policy updater param:{} has no name'.format(policy_updater_name))
                self._communicator.logger_info('Use policy updater name:{}'.format(policy_updater_name))
                policy_updater_info['param']['name'] = policy_updater_name
            param_name = policy_updater_info['param']['name']
            if policy_updater_name != param_name:
                self._communicator.logger_error(
                    'EvolutionML' + 'policy updater name:{} is not equal to param name:{}'.format(policy_updater_name,
                                                                                                  param_name))
                raise ValueError(
                    'policy updater name:{} is not equal to param name:{}'.format(policy_updater_name, param_name))

        ##########################################################
        # 创建DataModule
        self.data_module = DataModule(
            name_scope=task_name,
            communicator=communicator
        )

        ############################## 创建文件系统 ##############################
        self._file_system = DefaultFileSystem(
            project_name=task_name,
            data_module=self.data_module,
            communicator=communicator
        )  # 创建文件系统
        
        # 重新初始化DataModule
        self.data_module.init_data_module(file_system=self._file_system)

        # 获取更新策略的名称，加入scope
        scope_policy_map_dict = {}  # 用于保存文件系统的scope信息
        for policy_updater_name, policy_updater_info in policy_updater.items():
            unit_add_dic = self._file_system.add_scope(scope_name=policy_updater_name)  # 得到创建envinfo的信息
            scope_policy_map_dict[policy_updater_name] = unit_add_dic

        # 将scope_policy_map_dict添加到env_info中
        for policy_name, scope_env_inf in scope_policy_map_dict.items():
            self._env_info.add_element(
                **scope_env_inf
            )

            self._communicator.logger_success('EvolutionML:' + 'add scope:{} to env_info'.format(policy_name))

            self._scope_names_map_list.append(scope_env_inf['name'])
            self._scope_names_list.append(policy_name)

        ############################## 配置数据 ##############################
        # 创建env_info，并不创建unit
        # 获取候选动作策略大小
        self._candidate_action_num = len(real_policies)

        # 将候选策略信息添加到env info中
        # 创建候选动作策略
        # for action_name, action_shape in self._env_info.rl_action_dict.items():
        #     self._env_info.add_element(
        #         name='candidate_'+action_name,
        #         dtype=np.float32,
        #         shape=action_shape,
        #         size=self._candidate_action_num
        #     )
        #     self._candidate_action_name_list.append('candidate_'+action_name)
        #     self._communicator.logger_success('EvolutionML', 'add candidate action:{}'.format(action_name))

        # 为了方便加速开发，我们假设动作只有一个
        self._env_info.add_element(
            name='candidate_' + self._env_info.rl_action_name,
            dtype=np.float32,
            shape=self._env_info.rl_action_info.shape,
            size=self._candidate_action_num
        )

        self._candidate_action_name = 'candidate_' + self._env_info.rl_action_name
        self._communicator.logger_success(
            'EvolutionML:' + 'add candidate action:{}'.format(self._env_info.rl_action_name))

        # 添加state update到env info中
        self._env_info.add_element(
            name='state_update_flag',  # 这个名称是否可以改一下？
            dtype=np.int32,
            shape=(self._candidate_action_num,),
            size=1,
        )  # 该部分用于控制状态更新
        self._communicator.logger_success('EvolutionML:' + 'add state update flag')

        # 添加send flag到env info中
        self._env_info.add_element(
            name='send_flag',
            dtype=np.bool_,
            shape=(1,),
            size=1,
        )
        self._communicator.logger_success('EvolutionML:' + 'add send flag')

        # robot API control action 配置
        # for action_name, action_shape in self._env_info.rl_action_dict.items():
        #     self._env_info.add_element(
        #         name='control_'+action_name,
        #         dtype=np.float32,
        #         shape=action_shape,
        #         size=1
        #     )
        #     self._control_action_name_list.append('control_'+action_name)
        #     self._communicator.logger_success('EvolutionML', 'add control action:{}'.format(action_name))

        self._env_info.add_element(
            name='control_' + self._env_info.rl_action_name,
            dtype=np.float32,
            shape=self._env_info.rl_action_info.shape,
            size=1
        )
        self._control_action_name = 'control_' + self._env_info.rl_action_name
        self._communicator.logger_success(
            'EvolutionML:' + 'add control action:{}'.format(self._env_info.rl_action_name))

        # 添加control update flag到env info中
        self._env_info.add_element(
            name='control_update_flag',
            dtype=np.int32,
            shape=(1,),
            size=1
        )  # 该部分用于控制控制更新
        self._communicator.logger_success('EvolutionML:add control update flag')

        # 添加current state到env info中
        # TODO: current state请确定好名称，现在有点冲突
        for state_name, state_element in self._env_info.rl_state_dict.items():
            self._env_info.add_element(
                name='current_' + state_name,
                dtype=np.float32,
                shape=state_element.shape,
                size=1
            )
            self._current_state_name_list.append('current_' + state_name)
            self._current_state_name_map_list.append(state_name)
            self._data_names_in_buffer_list.append(state_name)
            self._communicator.logger_success('EvolutionML:add current state:{}'.format(state_name))

        for state_name, state_element in self._env_info.rl_state_dict.items():
            self._env_info.add_element(
                name='next_' + state_name,
                dtype=np.float32,
                shape=state_element.shape,
                size=1
            )
            self._next_state_name_list.append('next_' + state_name)
            self._next_state_name_map_list.append('next_' + state_name)
            self._data_names_in_buffer_list.append('next_' + state_name)
            self._communicator.logger_success('EvolutionML:add next state:{}'.format(state_name))

        # 添加candidate action flag到env info中
        self._env_info.add_element(
            name='candidate_action_flag',
            dtype=np.bool_,
            shape=(self._candidate_action_num,),
            size=1
        )

        for reward_name, _ in self._env_info.rl_reward_dict.items():
            self._data_names_in_buffer_list.append(reward_name)
            self._reward_name_list.append(reward_name)

        # TODO 以下需要改进
        self._data_names_in_buffer_list.append('action')
        self._data_names_in_buffer_list.append('mask')

        # for action_name, _ in self._env_info.rl_action_dict.items():
        #     self._data_names_in_buffer_list.append(action_name)
        #     self._action_name_list.append(action_name)
        ############################## 创建共享内存 ################################

        # 我们使用进程0创建共享内存
        if self._communicator.process_id == 0:
            self._communicator.logger_info('EvolutionML: create shared memory')
            self.data_module.read_data_unit_info_from_datainfo(self._env_info, exist=False)
            self._communicator.logger_info('Shared memory successfully created')
            self._communicator.barrier()
            self._communicator.logger_info('Waiting other process link and read shared memory...')
            self._communicator.barrier()

        # 剩余进程读取共享内存
        if self._communicator.process_id != 0:
            self._communicator.logger_info('Waiting main process create shared memory...')
            self._communicator.barrier()
            self._communicator.logger_info('EvolutionML: read shared memory')
            self.data_module.read_data_unit_info_from_datainfo(self._env_info, exist=True)
            self._communicator.logger_info('Share memory successfully linked')
            self._communicator.barrier()

        self._communicator.logger_success('All process read/create shared memory')

        ############################## 进程配置模块 ####################################
        # 此模块将在未来使用
        self._communicator.config_process_task_yaml(process_config_path)
        self._communicator.logger_success('EvolutionML: config process task')
        self._communicator.logger_warning('EvolutionML: process config module will be used in the future')

        # 完成此步骤以后，每个进程的资源

        ############################## 对DataModular定义接口部分 ####################################

        # 这一部分的操作主要是为了bind
        self.data_module.set_RobotAPI_state_unit_map(self._current_state_name_list, self._current_state_name_map_list)
        self.data_module.set_RobotAPI_control_action_unit(self._control_action_name)
        self.data_module.set_RobotAPI_next_state_unit_map(self._next_state_name_list, self._next_state_name_map_list)
        self.data_module.set_rewards_unit(self._reward_name_list)
        self.data_module.set_RobotAPI_state_update_flag_unit('state_update_flag')
        self.data_module.set_RobotAPI_control_update_flag_unit('control_update_flag')
        self.data_module.set_canidate_action_unit(self._candidate_action_name)

        self.data_module.set_history_number_map_inv(scope_names=self._scope_names_list,
                                                    map_names=self._scope_names_map_list)
        self.data_module.set_send_flag('send_flag')
        self.data_module.set_control_state_flag(rl_flag_dict['control_state_flag'])
        self.data_module.set_control_action_flag(rl_flag_dict['control_action_flag'])
        self.data_module.set_canidate_action_flag('candidate_action_flag')

        # self._communicator.logger_info('EvolutionML', 'set RobotAPI state and control unit')

        ############################## 分进程创建所需模块 ##############################
        # TODO: 暂时使用if else 未来采用自动化配置,合并在config_process_task_yaml中
        # TODO: 当前创建很麻烦后期需要优化

        # 0号节点负责policy_updater的创建
        if self._communicator.process_id == 0 or self._communicator.force_run:
            self._communicator.logger_info('EvolutionML: create policy updater.')

            # TODO: 这个地方需要改进，目前就支持一个policy_updater，优化policy updater的配置
            for policy_updater_name, policy_updater_info in policy_updater.items():
                # self._policy_updater = policy_updater_info['updater'](**policy_updater_info['param'], 
                #                                                       communicator=self._communicator,
                #                                                       data_module=self.data_module,
                #                                                       file_system=self._file_system
                #                                                       )

                self._policy_updater = RealUpdaterStarter(
                    policy_updater_name=policy_updater_name,
                    policy_updater=policy_updater_info['updater'],
                    policy_updater_param=policy_updater_info['param'],
                    capacity=self._capacity,
                    data_names_in_buffer=self._data_names_in_buffer_list,
                    offline_dataset_path=self._offline_dataset_path,
                    data_module=self.data_module,
                    communicator=self._communicator,
                    file_system=self._file_system
                )

        if self._communicator.process_id == 1 or self._communicator.force_run:
            self._communicator.logger_info('EvolutionML: create robot api.')
            for api_name, api_info in robot_api.items():
                self._robot_api = api_info['api'](**api_info['param'],
                                                  communicator=self._communicator,
                                                  data_module=self.data_module,
                                                  file_system=self._file_system
                                                  )

        if self._communicator.process_id == 2 or self._communicator.force_run:
            self._communicator.logger_info('EvolutionML: create policy selector.')
            for selector_name, selector_info in policy_selector.items():
                self._policy_selector = selector_info['selector'](**selector_info['param'],
                                                                  communicator=self._communicator,
                                                                  data_module=self.data_module,
                                                                  file_system=self._file_system
                                                                  )
        if self._communicator.process_id > 2 or self._communicator.force_run:
            self._communicator.logger_info('EvolutionML: create real policies.')

            # 获取总进程数
            total_process = self._communicator.total_process_num
            self._communicator.logger_info('EvolutionML: total process:{}'.format(total_process))

            new_id = self._communicator.process_id - 2
            i = 1

            for policy_name, policy_info in real_policies.items():
                if new_id == i:
                    self._real_policies = policy_info['policy'](**policy_info['param'],
                                                                communicator=self._communicator,
                                                                file_system=self._file_system,
                                                                data_module=self.data_module
                                                                )
                    break
                i += 1
        # TODO: 这里需要改进，现在这个地方需要靠if else来判断，后期需要优化。

        self._communicator.logger_success('EvolutionML: init EvolutionML complete.')

        self._communicator.barrier()

    def run(self):
        """
        用于运行EvolutionML。

        在运行的时候，这个函数会进行现实仿真，现实中外部环境是连续的，而不是离散的。
        因此会有两个循环，一个是外部环境的循环，一个是采样循环，不过这种一般是在支持实时仿真的情况下。

        交互形式的改进：
            一次交互产生, s,a,s',r这是一组完整的交互，读取进程会有明确的信号，交互过程中严格同步进行。

        """

        if self._communicator.process_id == 0 or self._communicator.force_run:
            self._communicator.logger_info('EvolutionML: start policy updater')
            self._policy_updater.run()

        if self._communicator.process_id == 1 or self._communicator.force_run:
            self._communicator.logger_info('EvolutionML: start robot api')
            self._robot_api.run()

        if self._communicator.process_id == 2 or self._communicator.force_run:
            self._communicator.logger_info('EvolutionML: start policy selector')
            self._policy_selector.run()

        if self._communicator.process_id > 2 or self._communicator.force_run:
            self._communicator.logger_info('EvolutionML: start real policies')
            self._real_policies.run()
