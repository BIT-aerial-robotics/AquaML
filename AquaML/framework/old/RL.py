'''
在线强化学习框架
'''
import AquaML
from AquaML.framework.FrameWorkBase import FrameWorkBase
from AquaML import logger, communicator, settings, data_module, file_system,recorder
from AquaML.worker import RLEnvBase as EnvBase
from AquaML.algo.RLAlgoBase import RLAlgoBase
from AquaML.param.ParamBase import RLParmBase
from AquaML.core.DataInfo import DataInfo

class RL(FrameWorkBase):
    """
    RL用于在线强化学习。
    该框架将支持以下功能：
    1. 单进程下Vec环境的在线强化学习。
    2. 多进程下Vec环境的在线强化学习。
    3. IsaacGym环境的在线强化学习。
    """
    
    def __init__(self,
                 env_class: EnvBase,
                 algo: RLAlgoBase,
                 hyper_params: RLParmBase,
                 model_dict: dict,
                 task_config_yaml:str=None,
                 ):
        """
        RL的构造函数。

        Args:
            env_class (EnvBase): 交互的环境的class。
            algo (RLAlgoBase): 强化学习算法。
            hyper_params (RLParmBase): 算法超参数。
            model_dict (dict): 模型字典。
            task_config_yaml (str): 任务配置文件。
        """
        
        super(RL, self).__init__()
        
        # 必须要初始化的参数
        self._env_class = env_class
        env = env_class(**hyper_params.envs_args)
        
        ##############################
        # 1. 配置每个进程的资源使用
        ##############################
        if task_config_yaml is not None:
            logger.info(f'Load task config from {task_config_yaml}')
            task_config_dict = settings.load_yaml(task_config_yaml)
        else:
            task_config_dict = {}
            task_config_dict['task_info'] = {
                'minimum_num_process':1,
                'GPU_required':'AUTO',
            }
            if communicator.size>1:
                # 多进程任务分配
                task_config_dict['updater'] = {
                    'process_id':0,
                    'GPU_enabled':'AUTO',
                    'GPU_id':'AUTO',
                }
                
                task_config_dict['collector'] = {
                    'process_id':'AUTO',
                    'GPU_enabled':False,
                    'GPU_id':False,
                }
            else:
                # 单进程任务分配
                task_config_dict['updater_collector'] = {
                    'process_id':0,
                    'GPU_enabled':'AUTO',
                    'GPU_id':'AUTO',
                }
            
        self.config_task(task=task_config_dict)
        
        ##############################
        # 2.初始化算法,初始化file_system，data_module
        ##############################
        
        self._algo:RLAlgoBase = algo(
            hyper_params=hyper_params,
            model_dict=model_dict,
        )
        self._hyper_params = hyper_params
        
        self.epoch = hyper_params.epoch
        
        # 配置对应模块
        self.config_file_system(algo_name=self._algo.algo_name)
        data_module.config_algo(algo_name=self._algo.algo_name)
        

        
        ##############################
        # 3. 获取该RL任务所需要的数据信息
        # 任务采取的数据格式为(num_env, step_num, ...)
        ##############################
        if communicator.size == 1:
            worker_num = 1
        else:
            worker_num = communicator.size - 1
        total_env_num = hyper_params.env_num*worker_num
        rollout_steps = hyper_params.rollout_steps
        
        # 获取动作模型的输出信息
        self._actor_output_info:DataInfo = self._algo.actor.output_info
        
        # 将算法中产生的多余数据信息添加到actor_output_info中
        self._actor_output_info = self._actor_output_info + self._algo.action_info 
        
        self._actor_output_info = self._actor_output_info + self._algo.algo_data_info
        
        self._obs_info = env.obs_info
        
        self._reward_info = env.reward_info
        
        ##############################
        # 4. 配置data_module
        ##############################
        
        # 创建next_obs_info
        self._next_obs_info = self._obs_info.insert_prefix('next_')
        
        # 创建terminal, truncated信息
        self._additional_info = DataInfo()
        self._additional_info.add_info('terminal', shape=(1,), dtype=bool)
        self._additional_info.add_info('truncated', shape=(1,), dtype=bool)        
        
        # 释放资源
        env.close()
        del env
        
        # self._algo:RLAlgoBase = algo(
        #     hyper_params=hyper_params,
        #     model_dict=model_dict,
        # )

        
        # 全局参数进行配置
        settings.set_env_num(total_env_num)
        settings.set_steps(rollout_steps) # 每个进程采样数
        
        
        # 多进程情况下使用共享内存，将数据注册到data_module中
        # TODO：这个部分移入worker当中
        if communicator.size > 1:
            unit_size = total_env_num
            if hyper_params.independent_model:
                # 使用独立模型时，将采用使用一个很大的buffer
                rollout_steps = hyper_params.rollout_steps
                self._actor_output_info.add_axis0_shape(rollout_steps)
                self._obs_info.add_axis0_shape(rollout_steps)
                self._next_obs_info.add_axis0_shape(rollout_steps)
                self._reward_info.add_axis0_shape(rollout_steps)
            
            # TODO: 未来优化data_infos的配置
            data_infos = (self._actor_output_info, self._obs_info,self._next_obs_info ,self._reward_info, self._additional_info)
        
            # 配置数据单元
            AquaML.config_data_units(
                data_infos=data_infos,
                units_name=self._algo.algo_name,
                size=unit_size,
                using_org_shape=True
            )
        
        # 单进程采用list存储数据
        else:
            data_infos = (self._actor_output_info, self._obs_info,self._next_obs_info ,self._reward_info,self._additional_info)

            AquaML.config_data_lists(
                data_infos=data_infos,
                lists_name=self._algo.algo_name,
                size=rollout_steps, # TODO：统一最后数据接口，最好由worker自己定义
            )
        
                 
        # 重新初始化算法
        self._algo.init()
        
        ##############################
        # 4. 配置worker
        ##############################
            
        if communicator.size == 1:
            # 使用单进程的RLWorker
            from AquaML.worker.RLWorker import RLWorker
            self._worker = RLWorker(
                # env=env,
                env_class=env_class,
                algo=self._algo,
                hyper_params=hyper_params,
                obs_names=self._obs_info.names,
                actor_output_names=self._actor_output_info.names,
                reward_names=self._reward_info.names
            )

    
    def run(self):
        """
        运行RL任务。
        """
        
        for epoch in range(self.epoch):
            logger.info(f'Epoch {epoch} start')
            data_dict = self._worker.roll()
            tracker = self._algo.train(data_dict)
            recorder.record_scalar(tracker.get_data())
        