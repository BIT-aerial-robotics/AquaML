from AquaML.Tool.IsaacGymMaker import IsaacGymMaker
from AquaML.Tool.AerialGymMaker import AerialGymMaker
from AquaML.worker.RLWorker import RLWorker
from AquaML.param.ParamBase import RLParmBase
from AquaML.worker.RLCollector import RLIsaacCollector
from AquaML.algo.RLAlgoBase import RLAlgoBase

class RLAerialGymWorker(RLWorker):
    """
    该worker适用于IsaacGym环境。
    """
    
    def __init__(self,
                 env_class: AerialGymMaker,
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
        
        super(RLAerialGymWorker, self).__init__(
            env_class=env_class,
            hyper_params=hyper_params,
        )
        
    def create_env(self,env_num:int,env_class:AerialGymMaker,envs_args: dict) -> None:
        """
        创建环境。
        
        Args:
            env_num (int): 环境的数量。
            env_class (EnvBase): 环境的类。
            envs_args (dict): 环境的参数。
        """
        
        self.env = env_class(
            env_name=envs_args['env_name'],
            env_num=env_num,
            # env_args=envs_args['env_args'],
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
        
        self.collector = RLIsaacCollector(
            algo_name=self.algo.algo_name,
            obs_names=self.obs_names,
            actor_output_names=self.actor_output_names,
            reward_names=self.reward_names,
            summary_steps=self.hyper_params.summary_steps,
        )
    
    def step(self)->bool:
        """
        采样一次数据。
        
        return:
            summary_flag: 是否需要进行汇总。
        """

        action_dict = self.algo.get_action(self.obs)
        
        
        next_obs, computing_obs, rewards, terminated, truncated, _ = self.env.auto_step(action_dict,max_step=self.max_step)
        
        summary_flag = self.collector.append(
            obs=self.obs,
            next_obs=next_obs,
            action=action_dict,
            reward=rewards,
            terminal=terminated,
            truncated=truncated
        )
        
        self.obs.update(computing_obs)

        # obs_, reward, done, info = self.env.step(action_dict['action'])

        # self.collector.append
        
        return summary_flag
   