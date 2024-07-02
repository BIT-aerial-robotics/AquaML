from AquaML.policy.PolicyBase import PolicyBase
from AquaML.core.old.DataModule import DataModule
from AquaML.core.old.FileSystem import FileSystemBase
from AquaML.communicator.CommunicatorBase import CommunicatorBase
import keras
import os

class FixNNPolicy(PolicyBase):
    
    def __init__(self,
                 name_scope:str,
                 candidate_action_id:int,
                 keras_model,
                 weight_path:str,
                 data_module:DataModule,
                 file_system:FileSystemBase,
                 communicator:CommunicatorBase,
    ):
        """
        不需要更新的策略，作为专家策略使用。
        
        这个类继承自PolicyBase，是一个不需要更新的策略，作为专家策略使用。
        
        Args:
            name_scope (str): 策略名称。
            candidate_action_id(int): 候选策略在数据中排序.
            keras_model (keras.Model): keras模型,传入类，不要具体实例化。只需要类。
            weight_path (str): 权重路径。
            data_module (DataModule): 数据模块。用于获取数据的shape等信息。由系统自动传入。
            file_system (FileSystemBase): 文件系统。用于文件的存储和读取。由系统自动传入。
            communicator (CommunicatorBase): 通信模块。用于多进程通信以及log等。由系统自动传入。
        """
        communicator.logger_info('FixNNPolicy: Init FixNNPolicy')
        
        super().__init__(
            name_scope=name_scope,
            data_module=data_module,
            file_system=file_system,
            communicator=communicator,
            candidate_action_id=candidate_action_id
        )
        
        # 初始化模型并加载权重
        self._actor:keras.Model = keras_model()
        self.initialize_model(self._actor)
        
        try:
            self._actor.load_weights(weight_path)
        except FileExistsError:
            communicator.logger_error('FixNNPolicy: Load weight failed.')
            raise FileExistsError('Load weight failed.')
        
        try:
            self._input_names = self._actor.input_names
        except AttributeError:
            communicator.logger_error('FixNNPolicy: Get input names failed.')
            raise AttributeError('Get input names failed.')
    
    def run(self,):
        """
        调用通用的run接口，用于运行策略。
        """
        # TODO: 这个地方需要改进，做成服务的方式。
        # 等待程序开始
        self._data_module.wait_program_start()
        
        while not self._data_module.get_program_end:

            # 等待状态信息就绪信号，获取状态信息，然后重置状态就绪信息
            self._data_module.wait_indexed_flag_to_value(
                flag=self._data_module.control_state_flag,
                index=self._candidate_action_id,
                value=True,
                time_out=self._communicator.wait_time_out,
                check_interval=self._communicator.check_time_interval
            )
            input_states = self.rec_state()
            self._communicator.logger_info('got state: {}'.format(input_states))
            self._communicator.logger_info('flag:{} is {}'.format(self._data_module.control_state_flag.name,
                                                                  self._data_module.control_state_flag.get_data()))
            self._data_module.control_state_flag[0][self._candidate_action_id] = False
            self._communicator.logger_info('flag:{} is {}'.format(self._data_module.control_state_flag.name,
                                                                  self._data_module.control_state_flag.get_data()))

            # 获取候选动作
            action = self._actor(*input_states) # 这个地方需要改进，需要根据输入的shape来进行输入。
            self._communicator.logger_info('RealWorldPolicyBase: Ready to send candidate action: {}'.format(action))

            # 等待 PEX 获取上一个候选动作，然后发送候选动作，并告诉 PEX 新的候选动作已就绪
            self._data_module.wait_indexed_flag_to_value(
                flag=self._data_module.candidate_action_flag,
                index=self._candidate_action_id,
                value=False,
                time_out=self._communicator.wait_time_out,
                check_interval=self._communicator.check_time_interval
            )
            self.send_candidate_action(action[0]) # 发送action给候选策略
            self._communicator.logger_info(
                'RealWorldPolicyBase: Send candidate_action: {}'.format(self._data_module.candidate_action.get_data()))
            self._data_module.candidate_action_flag[0][self._candidate_action_id] = True
            