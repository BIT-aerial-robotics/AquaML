import abc
from AquaML.worker import RLEnvBase as EnvBase
from AquaML.worker.RLVectorEnv import RLVectorEnv
from AquaML.algo.RLAlgoBase import RLAlgoBase
from AquaML.param.ParamBase import RLParmBase
from AquaML.worker.RLCollector import RLCollectorBase
import copy

class RLWorkerBase(abc.ABC):
    """
    RL worker的基类。
    
    在所有的worker初始化的时候，应该先创建Worker()。
    当data module配置好了之后，调用init()进行初始化。
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
        """
        
        ##############################
        # 1. 声明所需要的变量。
        # 没有初始化的变量必须在使用之前被初始化。
        ##############################
        # 注意初始化的参数
        self.env:RLVectorEnv = None # 系统将自动调用create_env()进行初始化，请实现create_env()。
        self.collector:RLCollectorBase = None # 在init()中初始化，请实现init()。
        self.algo:RLAlgoBase = None # 在init()中将参数传入，请实现init()。
        self.obs_names = None
        self.actor_output_names = None
        self.reward_names = None
        
        # 保存参数
        self.hyper_params = hyper_params
        self.max_step = hyper_params.max_step
   
        
        # 运算需要的中间变量
        self.obs = None
        self.reset_flag = True # 是否需要重置环境。
        self.epoch = 0
        
        ##############################
        # 2. 初始化部分参数
        ##############################
        
        # 初始化环境
        self.create_env(
            env_num=self.hyper_params.env_num,
            env_class=env_class,
            envs_args=self.hyper_params.envs_args
        )
        
        
    @abc.abstractmethod
    def init(self, 
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
        pass
        
    @abc.abstractmethod
    def create_env(self,env_class:EnvBase,env_num:int,envs_args:dict):
        """
        创建环境。
        """
        pass
    
    @abc.abstractmethod
    def config_data_module(self):
        """
        配置数据模块,不同类型的worker需要不同的数据模块。
        """
        pass

    
    def step(self)->bool:
        """
        采样一次数据。
        
        return:
            summary_flag: 是否需要进行汇总。
        """

        action_dict,mu = self.algo.get_action(self.obs)
        
        
        next_obs, computing_obs, rewards, terminated, truncated, _ = self.env.auto_step(action_dict,max_step=self.max_step)
        
        summary_flag = self.collector.append(
            obs=copy.deepcopy(self.obs),
            next_obs=copy.deepcopy(next_obs),
            action=copy.deepcopy(action_dict),
            reward=copy.deepcopy(rewards),
            terminal=copy.deepcopy(terminated),
            truncated=copy.deepcopy(truncated)
        )
        
        self.obs.update(computing_obs)

        # obs_, reward, done, info = self.env.step(action_dict['action'])

        # self.collector.append
        
        return summary_flag

    @abc.abstractmethod
    def roll(self)->dict:
        """
        进行一次rollout。
        
        return:
            dict: 包含了所有数据。
        """
        pass

    
    ##############################
    # 通用接口
    ##############################
    # @property
    # def obs_info(self):
    #     return self._obs_info

    # @property
    # def reward_info(self):
    #     return self._reward_info