import time

from AquaML.policy.PolicyBase import PolicyBase
from AquaML.core.old.DataModule import DataModule
from AquaML.core.old.FileSystem import FileSystemBase
from AquaML.communicator.CommunicatorBase import CommunicatorBase
from threading import Thread
import keras
import os
# class Policy(Thread,PolicyBase):
    
#     def __init__(self, 
#                  name_scope:str, 
#                  keras_model_class, 
#                  model_id, 
#                  ID_flag:IDFlag, 
#                  data_module:DataModule, 
#                  file_system:DefaultFileSystem):
        
#         Thread.__init__(self)
#         PolicyBase.__init__(self, name_scope)
        
#         self.add_keras_model('model', keras_model_class)
#         self.data_module = data_module
        
#         self.ID_flag = ID_flag
#         self.model_id = model_id
#         self.paired_model_id = 1 - model_id
#         # self.add_keras_model('model2', keras_model_class)
#         self.file_system = file_system
    
#     def load_weight(self, weight_path: str, model_name):
        
#         """
#         加载权重
        
#         Args:
#             weight_path (str): 权重路径
#             model_name (str): 模型名称
        
        
#         """
        
#         self._model_dict[model_name].load_weights(weight_path)
        
#     def run(self):
        
#         """
#         重载run方法，用于实现多线程。
#         当ID_flag和当前模型的ID相同时，该模型用于和现实中交互，其余模型加载最新的权重。
#         """
        
#         # self.ID_flag.set(self.ID_flag.get()+1)
#         # print('Policy', self.ID_flag.get())
#         # self._model_dict['model'+str(self.ID_flag.get())].fit(self._data_unit_dict['data_unit1'], self._data_unit_dict['data_unit2'])
#         # self.ID_flag.clear()
        
#         while True:
            
#             if self.ID_flag.get() == self.model_id:
#                 # 用于和现实中交互， 并且这又此部分能够切换模型
                
#                 # 获取数据
#                 input_data = self.data_module.get_data(self.model.input_name)
                
#                 # TODO:添加探索策略
#                 action = self.model(*input_data)
                
#                 # 将动作写入对应的buffer池里面方便其他进程调用
#                 # TODO:这个地方需要根据整体框架修改,真机交互动作指令同意
            
#             else:
#                 # 一直加载最新模型，加载协议多少次加载一次
#                 path = self.file_system.get_history_model_root_path # TODO: 请确定好接口继续修改
#                 self.load_weight(path, 'actor') # TODO: 确定一下名称接口
#                 # TODO: 请确定好加载几次，后面慢慢改
            
            
                
class  DeterminateRealWorldPolicy(PolicyBase):
    """
    用于和现实世界交互的策略基类。
    
    这类策略包含两个模型，模型1，2，两者持续交替，其中一个模型用于和现实世界交互，另一个模型用于加载最新的权重。
    
    该类继承自PolicyBase，是用于和现实世界交互的策略基类。
    """
    
    def __init__(self,
                 name_scope:str,
                 candidate_action_id:int,
                 keras_model,
                 weight_path: str,
                 data_module:DataModule,
                 file_system:FileSystemBase,
                 communicator:CommunicatorBase,
                 switch_times:int=5,
                 ):
        """
        # TODO: 文件同步机制需要完善.
        
        初始化策略基类,这里规定了（候选）策略接口。
        
        name_scope用于识别策略的名称，更新策略配对。policy的名称要和更新策略名称相同，用于配对。
        
        我们使用switch_times来控制1，2模型切换规则，每当某个模型交互达到指定次数切换另外一个模型

        Args:
            name_scope (str): 策略名称。
            candidate_action_id(int): 候选策略在数据中排序.
            keras_model (keras.Model): keras模型,传入类，不要具体实例化。只需要类。
            data_module (DataModule): 数据模块。用于获取数据的shape等信息。由系统自动传入。
            file_system (FileSystemBase): 文件系统。用于文件的存储和读取。由系统自动传入。
            communicator (CommunicatorBase): 通信模块。用于多进程通信以及log等。由系统自动传入。
            switch_times (int, optional): 切换模型的次数。默认为50。 Defaults to 50.
        """
        communicator.logger_info('RealWorldPolicyBase'+'Init RealWorldPolicyBase')
        super().__init__(
            name_scope=name_scope,
            candidate_action_id=candidate_action_id,
            data_module=data_module,
            file_system=file_system,
            communicator=communicator
        )
        
        # self._candidate_action_id = candidate_action_id
        self._switch_times = switch_times
        self._total_interaction_times: int = 0
        
        # 同时创建两个模型，actor1和actor2
        self._actor1 = keras_model()
        self._actor2 = keras_model()
        
        # 初始化模型
        self.initialize_model(self._actor1)
        self.initialize_model(self._actor2)
        
        # 获取模型输入信息
        self._input_names = self._actor1.input_names
        
        # 获取模型对应的算法更新地址，用于加载最新的权重
        scope_file_element = file_system.get_scope_file_element(name_scope)
        cache_path = scope_file_element.cache_path
        
        self._communicator.logger_info('RealWorldPolicyBase'+'{} get cache path:{}'.format(name_scope,cache_path))
        
        try:
            self._actor1.load_weights(cache_path+'/actor.h5')
            self._actor2.load_weights(cache_path+'/actor.h5')

            self._communicator.logger_success('RealWorldPolicyBase: Load actor weights')
        except FileExistsError:
            self._communicator.logger_error('RealWorldPolicyBase: Load actor weights error')
            #raise FileExistsError('Load actor weights error')
        except OSError:
            self._actor1.load_weights(weight_path)
            self._actor2.load_weights(weight_path)
            self._communicator.logger_info('RealWorldPolicyBase: Load initial actor weights ')
            #raise FileExistsError('Load actor weights error')

        # 获取模型对应的算法history_model路径
        self._history_model_path = scope_file_element.history_model_path
        self._communicator.debug_info('RealWorldPolicyBase: {} get history model path:{}'.format(name_scope,self._history_model_path))
        
        # 多线程标志
        self._switch_flag = 1 # 用于切换模型的标志
        # 只有交互的线程能够修改这个标志
        self._load_complete_flag = False # 用于标志加载权重是否完成
        self._current_weight_path = None
        
    def get_robot_input_state(self):
        """
        获取机器人状态。
        
        Returns:
            list: 机器人状态。
        """
        
        # TODO: 需要加入帧同步功能
        
        input_data = []
        
        for input_name in self._input_names:
            data = self._data_module.robot_state_dict[input_name]
            input_data.append(data)
            
        return input_data
    
    def get_weight_path(self):
        """
        获取权重路径。
        
        Returns:
            str: 权重路径。
        """
        
        # 获取当前模型对应的算法history_model serial number更新到哪里了
        history_model_serial_number = self._data_module.history_number_dict[self._name_scope]
        
        # 得到完整的路径
        weight_path = os.path.join(self._history_model_path, str(history_model_serial_number.get_data()[0][0]))
        weight_path = os.path.join(weight_path, 'actor.h5')
        
        self._communicator.logger_info('RealWorldPolicyBase: {} get weight path:{}'.format(self._name_scope,weight_path))
        self._communicator.logger_info('RealWorldPolicyBase: {} get weight path:{}'.format(self._name_scope,weight_path))
        return weight_path
    
    
    ######################################## 交互接口 ########################################
    
    def load_weight_thread(self):
        """
        加载权重的线程。
        
        当_switch_flag为1时，加载actor2的权重，当_switch_flag为2时，加载actor1的权重。
        
        该线程只能够读取_switch_flag，不能够修改。
        """
        
        self._communicator.logger_info('RealWorldPolicyBase: Load weight thread start')
        
        self._data_module.wait_program_start()
        
        # TODO: 暂时用while True，后面需要修改
        while not self._data_module.get_program_end:
            
            if self._load_complete_flag:
                continue

            if self._current_weight_path == self.get_weight_path():
                continue
            
            if self._switch_flag == 1:
                try:
                    weight_path = self.get_weight_path()
                    self._actor2.load_weights(weight_path)
                    self._current_weight_path = weight_path
                    self._communicator.logger_success('RealWorldPolicyBase: Load actor2 weights from {}'.format(weight_path))
                except OSError:
                    self._communicator.logger_info('RealWorldPolicyBase: Load actor2 weights, not found')
                
                self._load_complete_flag = True
            else:
                try:
                    weight_path = self.get_weight_path()
                    self._actor1.load_weights(weight_path)
                    self._current_weight_path = weight_path
                    self._communicator.logger_success('RealWorldPolicyBase: Load actor1 weights from {}'.format(weight_path))
                except OSError:
                    self._communicator.logger_info('RealWorldPolicyBase: Load actor1 weights, not found')

                self._load_complete_flag = True
            time.sleep(0.05)
                
    def predict_action_thread(self):
        """
        预测动作的线程。
        
        当_switch_flag为1时，actor1预测动作，当_switch_flag为2时，actor2预测动作。
        
        该线程只能够读取_switch_flag，不能够修改。
        """
        
        self._communicator.logger_info('RealWorldPolicyBase: Predict action thread start')
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
            input_data = self.rec_state()
            self._communicator.logger_info('got state: {}'.format(input_data))
            self._communicator.logger_info('flag:{} is {}'.format(self._data_module.control_state_flag.name,
                                                                  self._data_module.control_state_flag.get_data()))
            self._data_module.control_state_flag[0][self._candidate_action_id] = False
            self._communicator.logger_info('flag:{} is {}'.format(self._data_module.control_state_flag.name,
                                                                  self._data_module.control_state_flag.get_data()))

            # 获取候选动作
            if self._switch_flag == 1:
                action = self._actor1(*input_data)
                self._communicator.logger_info('RealWorldPolicyBase: actor1 predict action')
            else:
                action = self._actor2(*input_data)
                self._communicator.logger_info('RealWorldPolicyBase: actor2 predict action')
            self._communicator.logger_info('RealWorldPolicyBase: Ready to send candidate action: {}'.format(action))

            # 等待 PEX 获取上一个候选动作，然后发送候选动作，并告诉 PEX 新的候选动作已就绪
            self._data_module.wait_indexed_flag_to_value(
                flag=self._data_module.candidate_action_flag,
                index=self._candidate_action_id,
                value=False,
                time_out=self._communicator.wait_time_out,
                check_interval=self._communicator.check_time_interval
            )
            self.send_candidate_action(action[0])
            self._communicator.logger_info(
                'RealWorldPolicyBase: Send candidate_action: {}'.format(self._data_module.candidate_action.get_data()))
            self._data_module.candidate_action_flag[0][self._candidate_action_id] = True

            # 数据写入
            # TODO: 这个地方需要添加帧同步功能
            # self._data_module.candidate_action[self._candidate_action_id] = action

            # 设置控制权
            self._total_interaction_times += 1  # 交互次数加1

            if self._total_interaction_times % self._switch_times == 0 and self._load_complete_flag:
                self._switch_flag = 3 - self._switch_flag

                self._load_complete_flag = False

    def run(self):
        """
        
        运行task的接口部分，框架将自动调用此方法，这里需要去实现，代码逻辑。
        """
        
        self._communicator.logger_info('RealWorldPolicyBase: Run RealWorldPolicyBase')

        self.load_weight_thread_handle = Thread(target=self.load_weight_thread)
        self.load_weight_thread_handle.start()
        
        self.predict_action_thread_handle = Thread(target=self.predict_action_thread)
        self.predict_action_thread_handle.start()
        
        self._communicator.logger_success('RealWorldPolicyBase: Run RealWorldPolicyBase success')
        self._communicator.logger_info('RealWorldPolicyBase: Block main thread')
        
        self.load_weight_thread_handle.join()
        self.predict_action_thread_handle.join()
        
        
        
        
        
            
        
