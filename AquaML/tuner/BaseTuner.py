'''
BaseTuner class
基础版的tuner， 遍历法
此类是所有tuner的基类，所有tuner都应该继承此类， 并且提供基础tuner
'''

from AquaML.BaseClass import BaseAlgo
from AquaML.starter.RLTaskStarter import RLTaskStarter
from AquaML.tuner.TunerParam import TunerParam
import tensorflow as tf
import os
import numpy as np
from mpi4py import MPI


class BaseTuner(BaseAlgo):
    def __init__(self, 
                 algo_hyperparameter, 
                 model_dict: dict,
                 tuner: TunerParam, 
                 algo_type: str = 'RL', 
                 env=None,
                 seed: int = 0, 
                 GPU_ENABLE: bool = True):
        super().__init__()
        
        # 调整random seed时候，我们建议搜索random seed，其他参数不建议搜索

        # 默认采用多线程
        # 子任务暂时不支持多线程

        # 获取线程信息
        self.comm = MPI.COMM_WORLD
        self.thread_id = self.comm.Get_rank()
        self.total_threads = self.comm.Get_size()

        self.tuner = tuner
        
        self.algo_type = algo_type
        self.model_dict = model_dict
        
        # 配置GPU
        # 当线程数超过1时候，主线程不在启用GPU，只作为数据节点
        if GPU_ENABLE:
            # 记录sample thread
            if self.total_threads > 1:
                self.sample_thread = self.total_threads - 1
            else:
                self.sample_thread = 1
            # 如果有多个线程， 关闭主线程的GPU
            if self.thread_id == 0:
                if self.total_threads > 1:
                    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                    gpus = tf.config.list_physical_devices('GPU')
                    tf.config.experimental.set_memory_growth(gpus[0], True) 
            
            # 获取GPU数目
            gpus = tf.config.list_physical_devices('GPU')
            num_gpus = len(gpus)
            
            if self.thread_id > 0:
                tf.config.experimental.set_visible_devices(gpus[(self.thread_id - 1) % num_gpus], True)
            
            
                    

        # 统一random seed，如果random seed不作为搜索参数
        if self.tuner.seed_list is None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
            
            
        self.algo_hyperparameter = algo_hyperparameter
        
        if algo_type == 'RL':
            self.algo = RLTaskStarter
            self.env = env
        
    
    def run(self):
        while True:
            # 设置random seed
            if self.tuner.seed_list is not None:
                seed = self.tuner.seed_list[self.seed_pointer*self.sample_thread + self.thread_id]
                np.random.seed(seed)
                tf.random.set_seed(seed)
            
            # 算法超参数设置
            for key, value in self.tuner.hyper_param_search_dict.items():
                pointer = getattr(self, key + '_pointer')
                setattr(self.algo_hyperparameter, key, value[pointer*self.sample_thread + self.thread_id])
                pointer += 1
                setattr(self, key + '_pointer', pointer)
            
            # 运行算法
            if self.algo_type == 'RL':
                pass
    
    def init_search(self):
        
        # 确定搜索参数，创建搜索指针
        if self.tuner.seed_list is not None:
            self.seed_pointer = 0
            self.seed_max_pointer = int(len(self.tuner.seed_list) / self.sample_thread)
            
        for key, value in self.tuner.hyper_param_search_dict.items():
            # 设置专属的搜索指针和搜索范围
            setattr(self, key + '_pointer', 0)
            setattr(self, key + '_max_pointer', int(len(value) / self.sample_thread))