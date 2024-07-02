'''
离线强化学习参数
'''

from AquaML.param.old.ParamBase import ParamBase

class TD3BCParams(ParamBase):
    
    def __init__(self,
                 epoch:int,
                 update_times:int,
                 batch_size:int,
                 update_start:int,
                 sigma:float=0.2,
                 noise_clip_range:float=0.5,
                 tau=0.005,
                 alpha=0.2,
                 gamma=0.99,
                 delay_update:int=2,
                 model_save_interval=10,
                 ):
        super().__init__()
        
        self.epoch = epoch
        self.update_times = update_times
        self.batch_size = batch_size
        self.sigma = sigma
        self.noise_clip_range = noise_clip_range
        self.tau = tau
        self.alpha = alpha
        self.gamma = gamma
        self.delay_update = delay_update
        self.update_start = update_start
        self.model_save_interval = model_save_interval

        self._param_dict['epoch'] = self.epoch
        self._param_dict['update_times'] = self.update_times
        self._param_dict['batch_size'] = self.batch_size
        self._param_dict['sigma'] = self.sigma
        self._param_dict['noise_clip_range'] = self.noise_clip_range
        self._param_dict['tau'] = self.tau
        self._param_dict['alpha'] = self.alpha
        self._param_dict['gamma'] = self.gamma
        self._param_dict['delay_update'] = self.delay_update
        self._param_dict['update_start'] = self.update_start
        self._param_dict['model_save_interval'] = self.model_save_interval


class IQLParams(ParamBase):

    def __init__(self,
                 epoch: int,
                 update_times: int,
                 batch_size: int,
                 update_start: int,
                 sigma: float = 0.2,
                 noise_clip_range: float = 0.5,
                 tau=0.005,
                 expectile=0.7,
                 temperature=3,
                 gamma=0.99,
                 delay_update: int = 2,
                 model_save_interval=10,
                 ):
        super().__init__()

        self.epoch = epoch
        self.update_times = update_times
        self.batch_size = batch_size
        self.sigma = sigma
        self.noise_clip_range = noise_clip_range
        self.tau = tau
        self.expectile = expectile
        self.temperature = temperature
        self.gamma = gamma
        self.delay_update = delay_update
        self.update_start = update_start
        self.model_save_interval = model_save_interval

        self._param_dict['epoch'] = self.epoch
        self._param_dict['update_times'] = self.update_times
        self._param_dict['batch_size'] = self.batch_size
        self._param_dict['sigma'] = self.sigma
        self._param_dict['noise_clip_range'] = self.noise_clip_range
        self._param_dict['tau'] = self.tau
        self._param_dict['gamma'] = self.gamma
        self._param_dict['delay_update'] = self.delay_update
        self._param_dict['update_start'] = self.update_start
        self._param_dict['model_save_interval'] = self.model_save_interval
        self._param_dict['expectile'] = self.expectile
        self._param_dict['temperature'] = self.temperature
