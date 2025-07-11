import torch
from AquaML.learning.model.model_cfg import ModelCfg
from loguru import logger

from AquaML import coordinator

class Model(torch.nn.Module):

    def __init__(self, model_cfg: ModelCfg):
        super(Model, self).__init__()

        self.model_cfg = model_cfg
        self.device = self._select_device()
        
    
    def _select_device(self):
        if self.model_cfg.device == 'auto':
            return coordinator.get_device()
        elif self.model_cfg.device == 'cpu':
            return 'cpu'
        elif self.model_cfg.device.startswith('cuda'):
            if coordinator.validate_device(self.model_cfg.device):
                return self.model_cfg.device
            else:
                logger.warning(f"Device {self.model_cfg.device} is not available, using CPU instead")
                return 'cpu'
        else:
            raise ValueError(f"Invalid device: {self.model_cfg.device}")
    
    def init_state_dict(self):
        """Initialize lazy PyTorch modules' parameters.

        .. hint::

            Calling this method only makes sense when using models that contain lazy PyTorch modules
            (e.g. model instantiators), and always before performing any operation on model parameters.
        """
        
        for input_name in self.model_cfg.inputs_name:
            
            input_cfg = coordinator.get_
        