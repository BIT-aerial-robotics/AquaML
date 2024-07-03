'''
下一代的算法超参数基类。
'''
import abc
import numpy as np

class ParamBase(abc.ABC):
    """
    
    这个param能过自动将所具有的参数打包成字典。
    """
    
    def __init__(self,algo_name:str,checkpoints_store_interval:int=5):
        self.algo_name = algo_name
        self.checkpoints_store_interval = checkpoints_store_interval
        self._wandb_other_name = None
    
    def get_param_dict(self)->dict:
        """
        自动搜索该类中所有的参数，并将其打包成字典。
        """
        
        param_dict = {}
        
        for key in self.__dict__.keys():
            if key.startswith('_'):
                continue
            elif key == 'envs_args':
                continue
            else:
                param_dict[key] = self.__dict__[str(key)]
        
        return param_dict
    
    @property
    def wandb_other_name(self):
        return self._wandb_other_name
    
    
class RLParmBase(ParamBase):
    """
    RL算法的超参数基类。
    """
    
    def __init__(self,
                 rollout_steps:int,
                 epoch:int,
                 algo_name:str,
                 summary_steps:int=1000,
                 gamma=0.99,
                 env_num:int=1,
                 max_step:int=np.inf,
                 envs_args:dict={},
                 checkpoints_store_interval:int=5,
                 independent_model:bool=True,
                 ):
        """
        RL算法的超参数基类。
        
        Args:
            rollout_steps (int): 每次rollout的步数。
            epoch (int): 训练的epoch数。
            gamma (float): 折扣因子。
            independent_model (bool): 确认每个进程中是否有独立的模型。当为True时，每个进程中都有独立的模型。当为False时，所有进程共享一个模型。
        """
        super(RLParmBase, self).__init__(algo_name=algo_name,checkpoints_store_interval=checkpoints_store_interval)
        self.rollout_steps = rollout_steps
        self.epoch = epoch
        self.gamma = gamma
        self.env_num = env_num
        self.max_step = max_step
        self.envs_args = envs_args
        self.independent_model = independent_model
        self.summary_steps = summary_steps
        
        self._wandb_other_name = envs_args['env_name']
        
        # self.algo_name = algo_name
    

if __name__ == '__main__':
    
    class TestParam(ParamBase):
        
        def __init__(self,):
            self.a = 1
            self.b = 2
            self.c = 3
            
    test_param = TestParam()
    
    print(test_param.get_param_dict())