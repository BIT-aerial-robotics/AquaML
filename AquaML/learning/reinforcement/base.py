from AquaML import coordinator
from typing import Mapping, Optional, Union, Tuple, Dict, Any
from AquaML.learning.model import Model
import torch
import gymnasium
from loguru import logger
import os
import copy
import datetime
import collections
from packaging import version
import time
import yaml
from abc import ABC, abstractmethod

class Agent:
    """Base class for all AquaML agents"""

    def __init__(self,
            models: Mapping[str, Model],
            memory: Optional[Union[Any, Tuple[Any]]] = None,
            observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
            action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
            device: Optional[Union[str, torch.device]] = None,
            cfg: Optional[dict] = None,
        ):
        """Initialize base agent
        
        Args:
            models: Dictionary of models used by the agent
            memory: Memory buffer for storing transitions
            observation_space: Environment observation space
            action_space: Environment action space
            device: Device to use for computations
            cfg: Configuration dictionary
        """
        self.models = models
        self.memory = memory
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.cfg = cfg if cfg is not None else {}
        
        # Checkpoint management (inspired by SKRL)
        self.checkpoint_modules = {}
        
        # Handle both dict and dataclass configurations
        if hasattr(self.cfg, 'get'):
            # Dictionary configuration
            self.checkpoint_interval = self.cfg.get("experiment", {}).get("checkpoint_interval", 0)
            self.checkpoint_store_separately = self.cfg.get("experiment", {}).get("store_separately", False)
            directory = self.cfg.get("experiment", {}).get("directory", "")
            experiment_name = self.cfg.get("experiment", {}).get("experiment_name", "")
        else:
            # Dataclass configuration  
            self.checkpoint_interval = getattr(self.cfg, "checkpoint_interval", 0)
            self.checkpoint_store_separately = getattr(self.cfg, "store_separately", False)
            directory = getattr(self.cfg, "directory", "")
            experiment_name = getattr(self.cfg, "experiment_name", "")
            
        self.checkpoint_best_modules = {"timestep": 0, "reward": -(2**31), "saved": False, "modules": {}}
        
        # Experiment directory setup
        if not directory:
            directory = os.path.join(os.getcwd(), "runs")
        if not experiment_name:
            experiment_name = "{}_{}".format(
                datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"), self.__class__.__name__
            )
        self.experiment_dir = os.path.join(directory, experiment_name)
        
        # Tracking data for monitoring
        self.tracking_data = collections.defaultdict(list)
        self._track_rewards = collections.deque(maxlen=100)
        self._track_timesteps = collections.deque(maxlen=100)
        self._cumulative_rewards = None
        self._cumulative_timesteps = None
        
        # FileSystem integration - 初始化时获取，避免重复初始化
        self._file_system = None
        self._runner_name = None
        self._checkpoint_dir = None
        self._history_model_dir = None
        try:
            from AquaML import coordinator
            if coordinator.file_system_manager.file_system_exists() and coordinator.runner_manager.runner_exists():
                self._file_system = coordinator.getFileSystem()
                self._runner_name = coordinator.getRunner()
                self._checkpoint_dir = self._file_system.getCheckpointDir(self._runner_name)
                self._history_model_dir = self._file_system.queryHistoryModelPath(self._runner_name)
                logger.debug(f"FileSystem integrated: checkpoint_dir={self._checkpoint_dir}, history_model_dir={self._history_model_dir}")
        except Exception as e:
            logger.debug(f"FileSystem not available during agent initialization: {e}")
        
        logger.debug(f"Agent initialized with {len(models)} models")
        
    def _get_internal_value(self, _module: Any) -> Any:
        """Get internal module/variable state/value
        
        Args:
            _module: Module or variable
            
        Returns:
            Module/variable state/value
        """
        return _module.state_dict() if hasattr(_module, "state_dict") else _module
    
    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent
        
        This method should be called before the agent is used.
        It will create the checkpoints directory if needed.
        
        Args:
            trainer_cfg: Trainer configuration
        """
        trainer_cfg = trainer_cfg if trainer_cfg is not None else {}
        
        # Auto-configure checkpoint interval
        if self.checkpoint_interval == "auto":
            self.checkpoint_interval = int(trainer_cfg.get("timesteps", 0) / 10)
        
        # Create checkpoint directory using FileSystem
        if self.checkpoint_interval > 0:
            try:
                # Use coordinator's FileSystem for path management
                file_system = coordinator.getFileSystem()
                runner_name = coordinator.getRunner() if coordinator.runner_manager.runner_exists() else "default"
                file_system.getCheckpointDir(runner_name)
            except Exception:
                # Fallback to direct creation
                os.makedirs(os.path.join(self.experiment_dir, "checkpoints"), exist_ok=True)
    
    def track_data(self, tag: str, value: float) -> None:
        """Track data for monitoring
        
        Args:
            tag: Data identifier (e.g. 'Loss / policy loss')
            value: Value to track
        """
        self.tracking_data[tag].append(value)
    
    def write_checkpoint(self, timestep: int, timesteps: int) -> None:
        """Write checkpoint (modules) to disk
        
        The checkpoints are saved in the directory 'checkpoints' in the experiment directory.
        
        Args:
            timestep: Current timestep
            timesteps: Number of timesteps
        """
        if not self.checkpoint_modules:
            logger.warning("No checkpoint modules defined. Use register_checkpoint_module() to add modules.")
            return
            
        tag = str(timestep if timestep is not None else datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"))
        
        # Use pre-initialized checkpoint directory
        if self._checkpoint_dir:
            checkpoint_dir = self._checkpoint_dir
        else:
            # Fallback to old behavior
            checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save modules separately
        if self.checkpoint_store_separately:
            for name, module in self.checkpoint_modules.items():
                checkpoint_path = os.path.join(checkpoint_dir, f"{name}_{tag}.pt")
                torch.save(self._get_internal_value(module), checkpoint_path)
                logger.info(f"Saved checkpoint for {name} at {checkpoint_path}")
        else:
            # Save whole agent
            modules = {}
            for name, module in self.checkpoint_modules.items():
                modules[name] = self._get_internal_value(module)
            checkpoint_path = os.path.join(checkpoint_dir, f"agent_{tag}.pt")
            torch.save(modules, checkpoint_path)
            logger.info(f"Saved agent checkpoint at {checkpoint_path}")
        
        # 注意：最佳模型现在在update_best_checkpoint中立即保存，不再在这里处理
    
    def register_checkpoint_module(self, name: str, module: Any) -> None:
        """Register a module for checkpointing
        
        Args:
            name: Module name
            module: Module to register
        """
        self.checkpoint_modules[name] = module
        logger.debug(f"Registered checkpoint module: {name}")
    
    def save(self, path: str) -> None:
        """Save the agent to the specified path
        
        Args:
            path: Path to save the model to
        """
        if not self.checkpoint_modules:
            logger.warning("No checkpoint modules defined. Use register_checkpoint_module() to add modules.")
            return
            
        modules = {}
        for name, module in self.checkpoint_modules.items():
            modules[name] = self._get_internal_value(module)
        
        # Create directory if it doesn't exist
        # Use FileSystem for directory management
        try:
            file_system = coordinator.getFileSystem()
            file_system.ensureDir(os.path.dirname(path))
        except Exception:
            # Fallback to direct creation
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save(modules, path)
        logger.info(f"Agent saved to {path}")
    
    def load(self, path: str) -> None:
        """Load the agent from the specified path
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        
        # Load with proper device handling
        if version.parse(torch.__version__) >= version.parse("1.13"):
            modules = torch.load(path, map_location=self.device, weights_only=False)
        else:
            modules = torch.load(path, map_location=self.device)
        
        if isinstance(modules, dict):
            for name, data in modules.items():
                module = self.checkpoint_modules.get(name, None)
                if module is not None:
                    if hasattr(module, "load_state_dict"):
                        module.load_state_dict(data)
                        if hasattr(module, "eval"):
                            module.eval()
                        logger.debug(f"Loaded checkpoint for module: {name}")
                    else:
                        logger.warning(f"Module {name} doesn't support load_state_dict")
                else:
                    logger.warning(f"Cannot load the {name} module. The agent doesn't have such an instance")
        else:
            logger.error("Invalid checkpoint format")
            
        logger.info(f"Agent loaded from {path}")
    
    def update_best_checkpoint(self, timestep: int, reward: float) -> None:
        """Update best checkpoint if current reward is better and save immediately
        
        Args:
            timestep: Current timestep
            reward: Current reward
        """
        if reward > self.checkpoint_best_modules["reward"]:
            self.checkpoint_best_modules["timestep"] = timestep
            self.checkpoint_best_modules["reward"] = reward
            
            # 立即保存最佳模型到history_model目录，避免内存积累
            if self._history_model_dir and self.checkpoint_modules:
                try:
                    if self.checkpoint_store_separately:
                        for name, module in self.checkpoint_modules.items():
                            best_path = os.path.join(self._history_model_dir, f"best_{name}_{timestep}.pt")
                            torch.save(self._get_internal_value(module), best_path)
                            logger.info(f"Saved best checkpoint for {name} at {best_path}")
                    else:
                        modules = {k: self._get_internal_value(v) for k, v in self.checkpoint_modules.items()}
                        best_path = os.path.join(self._history_model_dir, f"best_agent_{timestep}.pt")
                        torch.save(modules, best_path)
                        logger.info(f"Saved best agent checkpoint at {best_path}")
                    
                    self.checkpoint_best_modules["saved"] = True
                    
                except Exception as e:
                    logger.warning(f"Failed to save best checkpoint: {e}")
                    self.checkpoint_best_modules["saved"] = False
            else:
                # 如果FileSystem不可用，则保存到内存中作为fallback
                self.checkpoint_best_modules["saved"] = False
                self.checkpoint_best_modules["modules"] = {
                    k: copy.deepcopy(self._get_internal_value(v)) for k, v in self.checkpoint_modules.items()
                }
                logger.warning("FileSystem not available, storing best checkpoint in memory")
            
            logger.info(f"New best checkpoint at timestep {timestep} with reward {reward}")
    
    def act(self, states: Dict[str, torch.Tensor], timestep: int, timesteps: int) -> Dict[str, torch.Tensor]:
        """Generate actions from policy
        
        Args:
            states: Dictionary of environment states
            timestep: Current timestep
            timesteps: Total timesteps
            
        Returns:
            Dictionary containing actions and other outputs
        """
        raise NotImplementedError("Subclasses must implement act method")
        
    def record_transition(self, 
                         states: Dict[str, torch.Tensor],
                         actions: Dict[str, torch.Tensor], 
                         rewards: torch.Tensor,
                         next_states: Dict[str, torch.Tensor],
                         terminated: torch.Tensor,
                         truncated: torch.Tensor,
                         infos: Any,
                         timestep: int,
                         timesteps: int):
        """Record environment transition
        
        Args:
            states: Current states
            actions: Actions taken
            rewards: Rewards received
            next_states: Next states
            terminated: Episode termination flags
            truncated: Episode truncation flags
            infos: Additional info
            timestep: Current timestep
            timesteps: Total timesteps
        """
        raise NotImplementedError("Subclasses must implement record_transition method")
        
    def post_interaction(self, timestep: int, timesteps: int):
        """Called after each environment interaction
        
        Args:
            timestep: Current timestep
            timesteps: Total timesteps
        """
        pass
        
    def pre_interaction(self, timestep: int, timesteps: int):
        """Called before each environment interaction
        
        Args:
            timestep: Current timestep
            timesteps: Total timesteps
        """
        pass
        
    def init(self, trainer_cfg: Optional[dict] = None):
        """Initialize agent with trainer configuration
        
        Args:
            trainer_cfg: Trainer configuration dictionary
        """
        if trainer_cfg is not None:
            self.trainer_cfg = trainer_cfg
        pass
        
    def set_running_mode(self, mode: str):
        """Set running mode for the agent
        
        Args:
            mode: Running mode ('train' or 'eval')
        """
        self.running_mode = mode
        if mode == "train":
            for model in self.models.values():
                model.train()
        elif mode == "eval":
            for model in self.models.values():
                model.eval()
        logger.debug(f"Agent set to {mode} mode")
        
    def track_data(self, key: str, value: Any):
        """Track data for logging/monitoring
        
        Args:
            key: Data key
            value: Data value
        """
        # This is a placeholder for tracking functionality
        # In a full implementation, this would integrate with logging/monitoring systems
        logger.debug(f"Tracking data: {key} = {value}")
        pass

