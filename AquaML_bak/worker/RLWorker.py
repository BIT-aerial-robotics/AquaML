from AquaML.worker import RLEnvBase as EnvBase
from AquaML.algo.RLAlgoBase import RLAlgoBase
from AquaML.param.ParamBase import RLParmBase
from AquaML.worker.RLWorkerBase import RLWorkerBase
from AquaML import logger, data_module,recorder,settings
from AquaML.worker.RLCollector import RLCollector
from AquaML.worker.RLVectorEnv import RLVectorEnv
import AquaML
import numpy as np
import copy

class RLWorker(RLWorkerBase):
    """
    该worker适用于单进程。
    
    TODO: 明确env_info的内容。
    """
    
    def __init__(self,
                 env_class: EnvBase,
                 hyper_params: RLParmBase,
                 ):
        
        """
        RLWorker的构造函数。
        
        Args:
            env_class (EnvBase): 环境的类，使用时不需要实例化。
            algo (RLAlgoBase): 算法。
            hyper_params (RLParmBase): 超参数。
            env_info (dict): 环境包含哪些数据。
        """
        
        super(RLWorker, self).__init__(
            env_class=env_class,
            hyper_params=hyper_params,
        )
    
    
    def create_env(self,env_num:int,env_class:EnvBase,envs_args):
        """
        创建环境。
        
        Args:
            env_num (int): 环境的数量。
            env_class (EnvBase): 环境的类。
            envs_args (dict): 环境的参数。
        """
        
        self.env = RLVectorEnv(
            env_num=env_num,
            env_class=env_class,
            envs_args=envs_args,
        )
        
    def config_data_module(self,data_infos:tuple, infos_name:str):
        """
        配置数据模块。
        
        Args:
            data_infos (tuple): 数据信息。
        """
        
        AquaML.config_data_lists(
                data_infos=data_infos,
                lists_name=infos_name,
                size=self.hyper_params.rollout_steps,
            )
    
    def init(self, 
                algo:RLAlgoBase,
                obs_names:tuple,
                actor_output_names:tuple,
                reward_names:tuple,
                ):
        """
        配置完data module之后，调用init()进行初始化。

        Args:
            obs_names (tuple): obs的名称。
            actor_output_names (tuple): actor输出的名称。
            reward_names (tuple): reward的名称。
        """
        self.obs_names = obs_names
        self.actor_output_names = actor_output_names
        self.reward_names = reward_names
        self.algo = algo
        
        self.collector = RLCollector(
            algo_name=self.algo.algo_name,
            obs_names=self.obs_names,
            actor_output_names=self.actor_output_names,
            reward_names=self.reward_names,
            summary_steps=self.hyper_params.summary_steps,
        )
   
    def roll(self):
        # TODO: 确定一下这个地方返回什么好。
        
        self.epoch += 1
        
        if self.reset_flag:
            computing_obs,_ = self.env.auto_reset()
            self.reset_flag = False
            self.obs = copy.deepcopy(computing_obs)

        self.collector.reset()
        
        
        
        for steps in range(self.hyper_params.rollout_steps):
            summary_flag = self.step()

            if summary_flag:
                current_step=((self.epoch-1)*self.hyper_params.rollout_steps+(steps+1))*settings.env_num
                recorder.record_scalar(data_dict=data_module.rl_dict,
                                       step=current_step,)
        
        current_step=((self.epoch-1)*self.hyper_params.rollout_steps+(steps+1))*settings.env_num

        return self.collector.get_data(), current_step
            
    
          