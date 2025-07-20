from typing import Any, Mapping, Tuple, Union, Dict

import gymnasium

import torch
from torch.distributions import Normal

from AquaML import coordinator
from AquaML.learning.model.base import Model
from AquaML.learning.model.model_cfg import ModelCfg


# speed up distribution construction by disabling checking
Normal.set_default_validate_args(False)


class GaussianMixin:
    def __init__(
        self,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20,
        max_log_std: float = 2,
        reduction: str = "sum",
        role: str = "",
    ) -> None:
        """Gaussian mixin model (stochastic model)

        This mixin class should be used together with the Model base class.
        The inheriting class must implement the compute method.

        :param clip_actions: Flag to indicate whether the actions should be clipped to the action space (default: ``False``)
        :type clip_actions: bool, optional
        :param clip_log_std: Flag to indicate whether the log standard deviations should be clipped (default: ``True``)
        :type clip_log_std: bool, optional
        :param min_log_std: Minimum value of the log standard deviation if ``clip_log_std`` is True (default: ``-20``)
        :type min_log_std: float, optional
        :param max_log_std: Maximum value of the log standard deviation if ``clip_log_std`` is True (default: ``2``)
        :type max_log_std: float, optional
        :param reduction: Reduction method for returning the log probability density function: (default: ``"sum"``).
                          Supported values are ``"mean"``, ``"sum"``, ``"prod"`` and ``"none"``. If "``none"``, the log probability density
                          function is returned as a tensor of shape ``(num_samples, num_actions)`` instead of ``(num_samples, 1)``
        :type reduction: str, optional
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :raises ValueError: If the reduction method is not valid

        Example::

            # define the model
            >>> import torch
            >>> import torch.nn as nn
            >>> from AquaML.learning.model.base import Model
            >>> from AquaML.learning.model.gaussian import GaussianMixin
            >>> from AquaML.learning.model.model_cfg import ModelCfg
            >>>
            >>> class Policy(GaussianMixin, Model):
            ...     def __init__(self, model_cfg: ModelCfg,
            ...                  clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
            ...         Model.__init__(self, model_cfg)
            ...         GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
            ...
            ...         self.net = nn.Sequential(nn.Linear(60, 32),  # assuming 60 observations
            ...                                  nn.ELU(),
            ...                                  nn.Linear(32, 32),
            ...                                  nn.ELU(),
            ...                                  nn.Linear(32, 8))  # assuming 8 actions
            ...         self.log_std_parameter = nn.Parameter(torch.zeros(8))
            ...
            ...     def compute(self, data_dict):
            ...         states = data_dict["states"]
            ...         mean_actions = self.net(states)
            ...         log_std = self.log_std_parameter.expand_as(mean_actions)
            ...         return {"mean_actions": mean_actions, "log_std": log_std}
            ...
            >>> # create model configuration
            >>> model_cfg = ModelCfg(device="cpu", inputs_name=["states"])
            >>> model = Policy(model_cfg)
            >>>
            >>> print(model)
            Policy(
              (net): Sequential(
                (0): Linear(in_features=60, out_features=32, bias=True)
                (1): ELU(alpha=1.0)
                (2): Linear(in_features=32, out_features=32, bias=True)
                (3): ELU(alpha=1.0)
                (4): Linear(in_features=32, out_features=8, bias=True)
              )
            )
        """

        self._g_clip_actions = clip_actions and isinstance(
            self.action_space, gymnasium.Space
        )

        if self._g_clip_actions:
            self._g_clip_actions_min = torch.tensor(
                self.action_space.low, device=self.device, dtype=torch.float32
            )
            self._g_clip_actions_max = torch.tensor(
                self.action_space.high, device=self.device, dtype=torch.float32
            )

        self._g_clip_log_std = clip_log_std
        self._g_log_std_min = min_log_std
        self._g_log_std_max = max_log_std

        self._g_log_std = None
        self._g_num_samples = None
        self._g_distribution = None

        if reduction not in ["mean", "sum", "prod", "none"]:
            raise ValueError("reduction must be one of 'mean', 'sum', 'prod' or 'none'")
        self._g_reduction = (
            torch.mean
            if reduction == "mean"
            else (
                torch.sum
                if reduction == "sum"
                else torch.prod if reduction == "prod" else None
            )
        )

    def compute(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute method to be implemented by the inheriting class

        This method should return a dictionary containing at least:
        - "mean_actions": mean actions from the neural network
        - "log_std": log standard deviations for the actions

        :param data_dict: Model inputs
        :type data_dict: Dict[str, torch.Tensor]
        :return: Dictionary containing mean_actions and log_std
        :rtype: Dict[str, torch.Tensor]
        :raises NotImplementedError: This method must be implemented by the inheriting class
        """
        raise NotImplementedError(
            "compute method must be implemented by the inheriting class"
        )

    def act(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Act stochastically in response to the state of the environment

        :param data_dict: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states (optional)
        :type data_dict: dict where the values are typically torch.Tensor

        :return: Model output dictionary containing:
                 - ``"actions"``: actions to be taken by the agent
                 - ``"log_prob"``: log of the probability density function
                 - ``"mean_actions"``: mean actions from the distribution
                 - Additional outputs from the compute method
        :rtype: Dict[str, torch.Tensor]

        Example::

            >>> # given a batch of sample states with shape (4096, 60)
            >>> outputs = model.act({"states": states})
            >>> print(outputs["actions"].shape, outputs["log_prob"].shape, outputs["mean_actions"].shape)
            torch.Size([4096, 8]) torch.Size([4096, 1]) torch.Size([4096, 8])
        """
        # map from states/observations to mean actions and log standard deviations
        outputs = self.compute(data_dict)
        mean_actions = outputs["mean_actions"]
        log_std = outputs["log_std"]

        # clamp log standard deviations
        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, self._g_log_std_min, self._g_log_std_max)

        self._g_log_std = log_std
        self._g_num_samples = mean_actions.shape[0]

        # distribution
        # TODO:这个地方能够提速
        self._g_distribution = Normal(mean_actions, log_std.exp())

        # sample using the reparameterization trick
        actions = self._g_distribution.rsample()

        # clip actions
        if self._g_clip_actions:
            actions = torch.clamp(
                actions, min=self._g_clip_actions_min, max=self._g_clip_actions_max
            )

        # log of the probability density function
        taken_actions = data_dict.get("taken_actions", actions)
        
        # If taken_actions is a dictionary, extract the actions
        if isinstance(taken_actions, dict):
            if "actions" in taken_actions:
                taken_actions = taken_actions["actions"]
            else:
                # Take the first value if no "actions" key
                taken_actions = list(taken_actions.values())[0]
        
        log_prob = self._g_distribution.log_prob(taken_actions)
        if self._g_reduction is not None:
            log_prob = self._g_reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        # update outputs with new values
        outputs["actions"] = actions
        outputs["log_prob"] = log_prob
        outputs["mean_actions"] = mean_actions

        return outputs
        

    def get_entropy(self, role: str = "") -> torch.Tensor:
        """Compute and return the entropy of the model

        :return: Entropy of the model
        :rtype: torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        Example::

            >>> entropy = model.get_entropy()
            >>> print(entropy.shape)
            torch.Size([4096, 8])
        """
        # TODO:进行提速
        if self._g_distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._g_distribution.entropy().to(self.device)

    def get_log_std(self, role: str = "") -> torch.Tensor:
        """Return the log standard deviation of the model

        :return: Log standard deviation of the model
        :rtype: torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        Example::

            >>> log_std = model.get_log_std()
            >>> print(log_std.shape)
            torch.Size([4096, 8])
        """
        return self._g_log_std.repeat(self._g_num_samples, 1)

    def distribution(self, role: str = "") -> torch.distributions.Normal:
        """Get the current distribution of the model

        :return: Distribution of the model
        :rtype: torch.distributions.Normal
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        Example::

            >>> distribution = model.distribution()
            >>> print(distribution)
            Normal(loc: torch.Size([4096, 8]), scale: torch.Size([4096, 8]))
        """
        return self._g_distribution


class GaussianModel(GaussianMixin, Model):
    """Gaussian Model combining GaussianMixin and Model base class
    
    This is a convenience class that combines the GaussianMixin with the Model base class.
    Users can inherit from this class and only need to implement the compute method.
    
    Example::
    
        >>> class PolicyModel(GaussianModel):
        ...     def __init__(self, model_cfg: ModelCfg):
        ...         super().__init__(model_cfg)
        ...         self.net = nn.Sequential(
        ...             nn.Linear(3, 64),
        ...             nn.ReLU(),
        ...             nn.Linear(64, 1)
        ...         )
        ...         self.log_std_parameter = nn.Parameter(torch.zeros(1))
        ...
        ...     def compute(self, data_dict):
        ...         states = data_dict["states"]
        ...         mean_actions = self.net(states)
        ...         log_std = self.log_std_parameter.expand_as(mean_actions)
        ...         return {"mean_actions": mean_actions, "log_std": log_std}
    """
    
    def __init__(self, model_cfg: ModelCfg, 
                 clip_actions: bool = False,
                 clip_log_std: bool = True,
                 min_log_std: float = -20,
                 max_log_std: float = 2,
                 reduction: str = "sum"):
        """Initialize GaussianModel
        
        Args:
            model_cfg: Model configuration
            clip_actions: Whether to clip actions to action space
            clip_log_std: Whether to clip log standard deviations
            min_log_std: Minimum log standard deviation value
            max_log_std: Maximum log standard deviation value
            reduction: Reduction method for log probability
        """
        Model.__init__(self, model_cfg)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        
        # Initialize log_std parameter (will be overridden in child classes)
        self.log_std_parameter = torch.nn.Parameter(torch.zeros(1))
        
    def compute(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute method to be implemented by inheriting classes
        
        This method should return a dictionary containing:
        - "mean_actions": mean actions from the neural network
        - "log_std": log standard deviations for the actions
        
        Args:
            data_dict: Model inputs
            
        Returns:
            Dictionary containing mean_actions and log_std
        """
        raise NotImplementedError("Subclasses must implement the compute method")
