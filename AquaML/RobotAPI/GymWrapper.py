'''

该模块主要用于测试算法的bug，以及对算法的性能进行评估。

用此文件规定最终的接口。
'''
from AquaML.RobotAPI.APIBase import APIBase
from AquaML.core.old.FileSystem import DefaultFileSystem
from AquaML.communicator.CommunicatorBase import CommunicatorBase
from AquaML.core.old.DataModule import DataModule
import gymnasium as gym
import numpy as np
import copy

class GymWrapper(APIBase):
    def __init__(self,
                 name:str,
                 gym_env_name:str,
                 data_module:DataModule,
                 file_system:DefaultFileSystem,
                 communicator:CommunicatorBase,
                 gym_env_param:dict={},
                 run_frequency:int=1,
                 max_step:int=1000,
                 ):
        """
        用于创建gym环境，用于算法测试。

        Args:
            name (str): 环境的名称，目前没有任何作用。
            gym_env_name (str): gym环境的名称。
            data_module (DataModule): 数据传输模块，用于和机器人交互。
            file_system (DefaultFileSystem): 文件系统。
            gym_env_param (dict, optional): gym环境的参数。 Defaults to {}.
            run_frequency (int, optional): 取数据的频率，和前面的定义不太一样。 Defaults to 2.
        """
        
        super().__init__(
            name=name,
            data_module=data_module,
            file_system=file_system,
            communicator=communicator,
            run_frequency=run_frequency
        )
        
        self._gym_env = gym.make(gym_env_name, **gym_env_param)
        
        self._step_count = 0 # 用于记录当前的步数。

        self._max_step = max_step
        
    
    ############################### 外部调用接口 #################################
    def run(self):
        """
        用于运行gym环境。
        
        在运行的时候，这个函数会进行现实仿真，现实中外部环境是连续的，而不是离散的。
        因此会有两个循环，一个是外部环境的循环，一个是采样循环，不过这种一般是在支持实时仿真的情况下。
        
        交互形式的改进：
            一次交互产生, s,a,s',r这是一组完整的交互，读取进程会有明确的信号，交互过程中严格同步进行。
            
        
        
        """
        
        self._data_module.wait_program_start() # 等待任务启动
        
        # 从data_module中读取程序运行状态
        # program_running_state = self._data_module.get_program_running_state()
        # start_flag = program_running_state[0]
        # end_flag = program_running_state[1]
        
        
        
        reset_flag = True
        
        # TODO: 这个地方检查
        while not self._data_module.get_program_end:
            
            if reset_flag:
                # 重置环境
                observation, info = self._gym_env.reset()
                
                states_dict = {
                    'env_obs': observation
                }
                self._step_count = 0

                reset_flag = False

            # 发送状态信息到current state
            self.send_state(states_dict)

            # 告知policy状态信息已就绪
            self._communicator.logger_info('control_state_flag is {}'.format(self._data_module.control_state_flag.get_data()))
            self._data_module.control_state_flag[0] = [True, True]
            self._communicator.logger_info('control_state_flag is {}'.format(self._data_module.control_state_flag.get_data()))

            # 等待 PEX 发送的控制动作就绪，获取动作，并重置控制动作就绪标志
            self._data_module.wait_for_flag_to_true(
                flag=self._data_module.control_action_flag,
                time_out=self._communicator.wait_time_out,
                check_interval=self._communicator.check_time_interval
            )
            self._communicator.logger_info('flag:{} is {}'.format(self._data_module.control_action_flag.name,
                                                                  self._data_module.control_action_flag.get_data()))
            action = self.rec_action()
            self._data_module.control_action_flag[0] = [False]
            self._communicator.logger_info('flag:{} is {}'.format(self._data_module.control_action_flag.name,
                                                                  self._data_module.control_action_flag.get_data()))

            # 执行动作，完成一次交互，数据为s,a,s',r ，发送s',r和mask
            next_observation, reward, terminated, truncated, info = self._gym_env.step(*action)
            next_states_dict = {
                'next_env_obs': next_observation
            }
            rewards_dict = {
                'env_reward': [reward]
            }
            self._step_count += 1
            if terminated or self._step_count >= self._max_step:
                reset_flag = True
            self.send_next_state(next_states_dict)
            self.send_reward(rewards_dict)
            self.send_mask(1 - terminated)

            # 更新状态信息到本循环中的状态信息字典
            states_dict['env_obs'] = next_states_dict['next_env_obs']

            # 此时data_module中s,a,s',r,mask都已准备就绪，告知buffer获取本次交互数据，等待buffer收集完成后，回到循环开头发送新的状态信息
            self._communicator.logger_info('flag:{} is {}'.format(self._data_module.send_flag.name,
                                                                  self._data_module.send_flag.get_data()))
            self._data_module.send_flag[0] = [True]
            self._communicator.logger_info('flag:{} is {}'.format(self._data_module.send_flag.name,
                                                                  self._data_module.send_flag.get_data()))
            self._data_module.wait_for_flag_to_false(
                        flag=self._data_module.send_flag,
                        time_out=self._communicator.wait_time_out,
                        check_interval=self._communicator.check_time_interval
                    )
            
              
    ############################### 内部调用接口 #################################
    
    def get_state(self, env_state:dict, rewards:dict, mask:bool):
        """
        用于获取环境的状态。
        
        Args:
            env_state (dict): 环境的状态。
            rewards (dict): 奖励。
            mask (bool): mask。
        """
        
        # 获取状态写入锁
        val = self._data_module.robot_state_update_flag.get_data()
        lenth = val.shape[0]
        
        if np.sum(val) == lenth:
            # 获取状态
            self._communicator.debug_info('GymWrapper get state lock.')
            
            # 将数据写入到data_module中
            # 将env_state写入到data_module中
            for data_name, data in env_state.items():
                self._data_module.robot_state_dict[data_name].set_data(data)
                
            # 将rewards写入到data_module中
            for data_name, data in rewards.items():
                self._data_module.robot_state_dict[data_name].set_data(data)
            
            # 将mask写入到data_module中
            self._data_module.mask.set_data(mask)
            
            # 重置状态写入锁,只有写入端才能重置锁。
            self._data_module.robot_state_update_flag.reset_zero()
        else:
            self._communicator.debug_info('GymWrapper get state lock failed.')
    
    def control(self)->np.ndarray:
        """
        用于控制环境。从data_module中获取控制信号，然后将其发送给机器人。
        
        return: 
            action (np.ndarray): 动作。 
        """
        
        # 直接获取不考虑锁的问题。
        
        action = self._data_module.robot_control_action.get_data()
        
        return action
    
    def send_state(self, states:dict):
        """
        用于发送状态。
        
        从这个函数开始，简化所的设计锁，锁只有一个。

        Args:
            states (dict): 状态。
        """
        
        for state_name, state in states.items():
            self._data_module.robot_state_dict[state_name].set_data(state)
            
    def send_next_state(self, next_states:dict):
        """
        用于发送下一个状态。
        
        Args:
            next_states (dict): 下一个状态。
        """
        
        for state_name, state in next_states.items():
            self._data_module.robot_next_state_dict[state_name].set_data(state)
        
            
    def send_reward(self, rewards:dict):
        """
        用于发送奖励。
        
        Args:
            rewards (dict): 奖励。
        """
        
        for reward_name, reward in rewards.items():
            self._data_module.rewards_dict[reward_name].set_data(reward)
            
    def send_mask(self, mask:bool):
        
        self._data_module.env_mask.set_data([mask])
        
    # def send_action(self, action:np.ndarray):
    #     """
    #     用于发送动作。
        
    #     Args:
    #         action (np.ndarray): 动作。
    #     """
        
    #     self._data_module.robot_control_action.set_data(action)
        
    def rec_action(self):
        """
        用于接收动作。
        
        return:
            action (np.ndarray): 动作。
        """
        
        action = self._data_module.robot_control_action.get_data()
        
        return action
        