import torch
from loguru import logger
from packaging import version
import gymnasium

from AquaML import coordinator
from AquaML.learning.model.model_cfg import ModelCfg
from AquaML.data.base_unit import BaseUnit
from AquaML import coordinator

from typing import Dict, Any, Optional


class Model(torch.nn.Module):

    def __init__(self, model_cfg: ModelCfg):
        super(Model, self).__init__()

        self.model_cfg = model_cfg
        self.device = self._select_device()

        self.action_space: gymnasium.Space = None # type: ignore
        self.observation_space: gymnasium.Space = None # type: ignore

    def _select_device(self):
        if self.model_cfg.device == "auto":
            return coordinator.get_device()
        elif self.model_cfg.device == "cpu":
            return "cpu"
        elif self.model_cfg.device.startswith("cuda"):
            if coordinator.validate_device(self.model_cfg.device):
                return self.model_cfg.device
            else:
                logger.warning(
                    f"Device {self.model_cfg.device} is not available, "
                    f"using CPU instead"
                )
                return "cpu"
        else:
            raise ValueError(f"Invalid device: {self.model_cfg.device}")

    def set_action_space(self, action_space: gymnasium.Space):
        self.action_space = action_space

    def set_observation_space(self, observation_space: gymnasium.Space):
        self.observation_space = observation_space

    def init_state_dict(self):
        """Initialize lazy PyTorch modules' parameters.

        .. hint::

            Calling this method only makes sense when using models that
            contain lazy PyTorch modules (e.g. model instantiators), and
            always before performing any operation on model parameters.
        """

        data_dict: dict[str, torch.Tensor] = {}
        for input_name in self.model_cfg.inputs_name:
            data_unit:BaseUnit= coordinator.getDataUnit(input_name)

            raw_data = data_unit.getVirtualData()   

            data_dict[input_name] = raw_data

        if self.model_cfg.concat_dict:
            data_dict = coordinator.tensor_tool.concat_dict_tensors(data_dict)


    
    def forward(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass of the model

        This method calls the ``.act()`` method and returns its outputs

        :param data_dict: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
        :type data_dict: dict where the values are typically torch.Tensor

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is the log of the probability density function for stochastic models
                 or None for deterministic models. The third component is a dictionary containing extra output values
        :rtype: tuple of torch.Tensor, torch.Tensor or None, and dict
        """
        return self.act(data_dict)

        
    def compute(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Define the computation performed (to be implemented by the inheriting classes) by the models

        :param data_dict: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
        :type data_dict: dict where the values are typically torch.Tensor

        :raises NotImplementedError: Child class must implement this method

        :return: Computation performed by the models
        :rtype: tuple of torch.Tensor and dict
        """

        raise NotImplementedError("Child class must implement this method")


    def act(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:       
        """Act according to the specified behavior (to be implemented by the inheriting classes)

        Agents will call this method to obtain the decision to be taken given the state of the environment.
        This method is currently implemented by the helper models (**GaussianModel**, etc.).
        The classes that inherit from the latter must only implement the ``.compute()`` method

        :param data_dict: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
        :type data_dict: dict where the values are typically torch.Tensor

        :raises NotImplementedError: Child class must implement this method

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is the log of the probability density function for stochastic models
                 or None for deterministic models. The third component is a dictionary containing extra output values
        :rtype: tuple of torch.Tensor, torch.Tensor or None, and dict
        """
        raise NotImplementedError("The action to be taken by the agent (.act()) is not implemented")

    
    def save(self, path: str, state_dict: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """Save the model to the specified path

        :param path: Path to save the model to
        :type path: str
        :param state_dict: State dictionary to save (default: None).
                           If None, the model's state_dict will be saved
        :type state_dict: dict, optional

        Example::

            # save the current model to the specified path
            >>> model.save("/tmp/model.pt")
            
            # save an older version of the model to the specified path
            >>> old_state_dict = copy.deepcopy(model.state_dict())
            >>> # ...
            >>> model.save("/tmp/model.pt", old_state_dict)
        """
        try:
            # Create directory using FileSystem
            try:
                file_system = coordinator.getFileSystem()
                file_system.ensureDir(os.path.dirname(path))
            except Exception:
                # Fallback to direct creation
                import os
                os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save state dict
            torch.save(self.state_dict() if state_dict is None else state_dict, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model to {path}: {e}")
            raise

    def load(self, path: str) -> None:
        """Load the model from the specified path

        The final storage device is determined by the constructor of the model

        :param path: Path to load the model from
        :type path: str
        
        Example::

            # load the model onto the CPU
            >>> model = Model(model_cfg)
            >>> model.load("model.pt")

            # load the model onto the GPU 
            >>> model = Model(model_cfg)  # with device config pointing to GPU
            >>> model.load("model.pt")
        """
        try:
            import os
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
            
            # Load with proper device handling
            if version.parse(torch.__version__) >= version.parse("1.13"):
                state_dict = torch.load(path, map_location=self.device, weights_only=False)  # prevent torch:FutureWarning
            else:
                state_dict = torch.load(path, map_location=self.device)
                
            self.load_state_dict(state_dict)
            self.eval()
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")
            raise

    
    