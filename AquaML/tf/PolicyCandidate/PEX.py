import time

import tensorflow as tf
from AquaML.algo_base.PolicyCandidateBase import PolicyCandidateBase
from AquaML.communicator.CommunicatorBase import CommunicatorBase
from AquaML.core.old.DataModule import DataModule
from AquaML.core.old.FileSystem import FileSystemBase
from AquaML.param.PolicyCandidate import PEXParams
from AquaML.core.Tool import LossTracker
from AquaML.buffer.BufferBase import BufferBase
from AquaML.core.DataUnit import DataUnit
import numpy as np
import os
from threading import Thread

class PEX(PolicyCandidateBase):
    
    def __init__(self, 
                 critic1,
                 critic2,
                 initial_weight_path:str,
                 hyper_params:PEXParams,
                 communicator:CommunicatorBase,
                 data_module:DataModule,
                 file_system:FileSystemBase,
                 name:str="PEX",
                 ):
        """
        Policy Expansion for offline-to-online reinforcement learning.
        
        该算法基于PEX进行修改，此部分只包含PEX的算法选择部分，部分情况包含策略选择更新。
        
        Reference:
        [1] Zhang H , Xu W , Yu H .Policy Expansion for Bridging Offline-to-Online Reinforcement Learning[J].
            arXiv e-prints, 2023.

        Args:
            critic (tf.keras.Model): critic模型,不要具体实例化。只需要类。
            weight_path (str): critic模型的权重路径。
            hyper_params (PEXParams): PEX算法参数。
            communicator (CommunicatorBase): 通信模块。用于多进程通信以及log等。由系统自动传入。
            data_module (DataModule): 数据模块。用于获取数据的shape等信息。由系统自动传入。
            file_system (FileSystem): 文件系统。用于文件的存储和读取。由系统自动传入。
            buffer (BufferBase): buffer模块。用于存储数据。由系统自动传入。
            name(str): 算法的名称。
        """
        super().__init__(name, communicator, data_module, file_system)
        self._communicator.logger_info('PEX: Create PEX')
        
        self._critic1 = critic1()
        self._critic2 = critic2()
        self._initial_weight_path = initial_weight_path
        self._hyper_params = hyper_params

        self.load_interval = 5

        self._input_names = self._critic1.input_names
        
        self._candidate_action_num = self._data_module.candidate_action_flag.shape[0]

        self._switch_flag = 1  # 用于切换模型的标志
        self._load_complete_flag = True # 用于标志加载权重是否完成
        self._total_interaction_times = 1
        self._current_weight_path = None

        # 检查超参数是否合法    
        eps = self._hyper_params.eps
        if eps < 0 or eps > 1:
            self._communicator.logger_error('PEX: eps should be in [0,1]')
            raise ValueError('eps should be in [0,1]')

    def init(self, buffer):
        pass

    def init1(self,
             # buffer:BufferBase
             ):
        """
        初始化PEX算法。
        
        Args: 
            buffer (BufferBase): buffer模块。用于存储数据。由系统自动传入。
        """

        self._communicator.logger_info('PEX: Init PEX')

        # 初始化网络模型
        self.initialize_network(self._critic1)
        self.initialize_network(self._critic2)

        # 加载权重
        if self._hyper_params.algo == 'TD3BC':
            try:
                self._critic1.load_weights(self._initial_weight_path + '/q_critic1.h5')
                self._communicator.logger_info('PEX: Load critic1 weights')
            except FileExistsError:
                self._communicator.logger_error('PEX: Load critic1 weights error')
            try:
                self._critic2.load_weights(self._initial_weight_path + '/q_critic2.h5')
                self._communicator.logger_info('PEX: Load critic2 weights')
            except FileExistsError:
                self._communicator.logger_error('PEX: Load critic2 weights error')
        elif self._hyper_params.algo == 'IQL':
            try:
                self._critic1.load_weights(self._initial_weight_path + '/q_critic.h5')
                self._critic2.load_weights(self._initial_weight_path + '/q_critic.h5')
                self._communicator.logger_info('PEX: Load critic weights')
            except FileExistsError:
                self._communicator.logger_error('PEX: Load critic weights error')
            
    @tf.function
    def get_min_q_value(self, state, action):
        """
        获取Q值。
        
        获取最小的q值。
        
        
        Args:
            state (list): 输入的state。
            action (np.ndarray): 输入的action。
        """
        
        q1 = self._critic1(*state, action)
        q2 = self._critic2(*state, action)
        
        min_q = tf.minimum(q1, q2)
        
        return min_q

    def select_action(self):
        """
        根据输入的state选择action。
        
        这里选择方法是通过PEX给出的。

        选择好的action会送入data_module中。
        """
        # 获取state
        state_dict = self._data_module.robot_state_dict
        
        # 获取candidate action
        candidate_action = self._communicator.candidate_action
        
        input_state = self.get_corrosponding_state(state_dict, self._critic.input_names)
        
        
        min_q = self.get_min_q_value(input_state,candidate_action)
        
        logits = min_q * self._hyper_params.inv_temperature
        
        greedy_action = tf.argmax(logits, axis=1)
        
        if self._hyper_params.eps == 0:
            action = greedy_action
        elif self._hyper_params.eps > 0:
            # 随机选择action
            sample_action = tf.random.categorical(logits, 1)
            greedy_mask = tf.random.uniform(tf.shape(sample_action)) < self._hyper_params.eps
            action = tf.where(greedy_mask, sample_action, greedy_action)
            
        # 选择好的action会送入data_module中。
        self._data_module.robot_control_action.set_data(action)

    def get_weight_path(self):
        """
        获取权重路径。

        Returns:
            str: 权重路径。
        """
        # 获取模型对应的算法更新地址，用于加载最新的权重
        scope_file_element = self._file_system.get_scope_file_element(self._hyper_params.algo)

        # 获取当前模型对应的算法history_model serial number更新到哪里了
        history_model_serial_number = self._data_module.history_number_dict[self._hyper_params.algo]

        history_model_path = scope_file_element.history_model_path


        # 得到路径
        weight_path = os.path.join(history_model_path, str(history_model_serial_number.get_data()[0][0]))

        self._communicator.debug_info(
            'RealWorldPolicyBase: pex get weight path:{}'.format(weight_path))
        self._communicator.logger_info(
            'RealWorldPolicyBase: pex get weight path:{}'.format(weight_path))
        return weight_path

    def load_weight_thread(self):
        """
        加载权重的线程。

        当_switch_flag为1时，加载actor2的权重，当_switch_flag为2时，加载actor1的权重。

        该线程只能够读取_switch_flag，不能够修改。
        """

        self._communicator.logger_info('PEX: Load weight thread start')

        self._data_module.wait_program_start()

        # TODO: 暂时用while True，后面需要修改
        while not self._data_module.get_program_end:

            if self._load_complete_flag:
                continue

            if self._current_weight_path == self.get_weight_path():
                self._communicator.logger_success('1PEX: {} {}'.format(self._current_weight_path, self.get_weight_path()))
                continue
            # self._communicator.logger_success('2PEX: {} {}'.format(self._current_weight_path, self.get_weight_path()))

            if self._hyper_params.algo == 'TD3BC':
                pass
                # try:
                #     weight_path = self.get_weight_path()
                #     weight_path1 = os.path.join(weight_path, 'q_critic1.h5')
                #     weight_path2 = os.path.join(weight_path, 'q_critic2.h5')
                #     self._critic1.load_weights(weight_path1)
                #     self._critic2.load_weights(weight_path2)  # TODO 从history？？？
                #     self._communicator.debug_info('PEX: Load critic weights')
                # except OSError:
                #     self._communicator.logger_info('PEX: Load critic weights, not found')
                #
                # self._load_complete_flag = True
            elif self._hyper_params.algo == 'IQL':

                if self._switch_flag == 1:
                    try:
                        weight_path = self.get_weight_path()
                        weight_path = os.path.join(weight_path, 'q_critic.h5')
                        self._critic2.load_weights(weight_path)
                        self._current_weight_path = weight_path
                        # self._communicator.logger_success('PEX: Load critic2 weights from {}'.format(weight_path))
                    except OSError:
                        self._communicator.logger_info('PEX: Load critic2 weights, not found')

                    self._load_complete_flag = True
                else:
                    try:
                        weight_path = self.get_weight_path()
                        weight_path = os.path.join(weight_path, 'q_critic.h5')
                        self._critic1.load_weights(weight_path)
                        self._current_weight_path = weight_path
                        # self._communicator.logger_success('PEX: Load critic1 weights from {}'.format(weight_path))
                    except OSError:
                        self._communicator.logger_info('PEX: Load critic1 weights, not found')

                    self._load_complete_flag = True
                time.sleep(0.05)
    
    def select_action_thread(self):
        self.init1()

        self._data_module.wait_program_start()
        
        while not self._data_module.get_program_end:
            
            # 等待所有candidate action生成完毕
            self._data_module.wait_whole_flag_to_value(
                flag=self._data_module.candidate_action_flag,
                value=True,
                time_out=self._communicator.wait_time_out,
                check_interval=self._communicator.check_time_interval
            )
            
            # 获取数据，包括状态和2个候选动作
            inputs = self.rec_inputs()
            self._communicator.logger_info('PEX: input_states: {}'.format(inputs))
            # 计算合理的q值
            if self._hyper_params.algo == 'TD3BC':
                q1 = self._critic1(inputs[0], inputs[1])
                q2 = self._critic2(inputs[0], inputs[1])
            elif self._hyper_params.algo == 'IQL':
                if self._switch_flag == 1:
                    q1, q2 = self._critic1(inputs[0], inputs[1])
                else:
                    q1, q2 = self._critic2(inputs[0], inputs[1])

            min_q = tf.minimum(q1, q2)

            # logits = min_q * self._hyper_params.inv_temperature

            selected_action_id = tf.argmax(min_q) # TODO:这个地方不太好不太对后面再改

            # q1 = tf.minimum(self._critic1(inputs), self._critic2(inputs))
            # q2 = tf.minimum(self._critic1(inputs), self._critic2(inputs))
            # selected_action_id = tf.argmax([q1, q2], 0)

            self._communicator.logger_info('PEX: control_action_id: {}'.format(selected_action_id))

            # 得到选择好的动作
            control_action = self._data_module.candidate_action[selected_action_id]
            # 发送选择好的动作
            self._data_module.robot_control_action.set_data(control_action)
            # 告知 env 动作已就绪
            self._communicator.logger_info('flag:{} is {}'.format(self._data_module.control_action_flag.name,
                                                                  self._data_module.control_action_flag.get_data()))
            self._data_module.control_action_flag[0] = True
            self._communicator.logger_info('flag:{} is {}'.format(self._data_module.control_action_flag.name,
                                                                  self._data_module.control_action_flag.get_data()))
            # 重置候选动作就绪标志
            self._data_module.candidate_action_flag.reset_false()

            self._total_interaction_times += 1

            if self._total_interaction_times % self.load_interval == 0 and self._load_complete_flag:
                self._switch_flag = 3 - self._switch_flag
                self._load_complete_flag = False

    def run(self):
        self._communicator.logger_info('PEX: Run ')

        self.load_weight_thread_handle = Thread(target=self.load_weight_thread)
        self.load_weight_thread_handle.start()

        self.select_action_thread_handle = Thread(target=self.select_action_thread)
        self.select_action_thread_handle.start()

        self._communicator.logger_success('PEX: Run  success')
        self._communicator.logger_info('PEX: ')

        self.load_weight_thread_handle.join()
        self.select_action_thread_handle.join()

    def rec_inputs(self):
        """
        用于接收输入。
        
        并且返回该策略所需要的输，以list返回。
        
        Returns:
            list: 返回该策略所需要的输。
        """
        
        # 接受state
        state_dict = self._data_module.robot_state_dict
        
        new_state_dict = {}
        
        # 匹配state
        
        for data_name, data in state_dict.items():
            
            new_data = np.ones(
                shape=(self._candidate_action_num, *data.shape),
                dtype=np.int32
            ) * data.get_data() # 扩充数据
            
            new_state_dict[data_name] = new_data
            
            
        
        # 接受candidate action
        candidate_actions = self._data_module.candidate_action
        
        new_state_dict['action'] = candidate_actions.get_data()
        
        input_list = []
        
        for input_name in self._input_names:
            input_list.append(new_state_dict[input_name])
        
        return input_list
    
    def optimize(self, buffer):
        pass