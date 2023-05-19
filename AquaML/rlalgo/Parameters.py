import abc
from AquaML.BaseClass import BaseParameter


class BaseParameter(BaseParameter, abc.ABC):

    def __init__(self, epoch_length: int,
                 n_epochs: int,
                 buffer_size: int,
                 batch_size: int,
                 update_interval: int = 0,
                 store_model_times: int = 5,
                 update_times=1,
                 eval_episodes=0,
                 batch_trajectory=False,
                 action_space_type: str = None,
                 eval_interval: int = 1,
                 ):
        """
        Parameters of environment.

        All the algorithm parameters should inherit this class.

        Args:
            epoch_length (int): The length of epoch.
            n_epochs (int): Times of optimizing the network.
            batch_size (int): batch size.
            update_interval (int): update interval.
            update_times (int): times for optimizing the network.
            batch_trajectory (bool): whether to use batch trajectory. This argument is to
            control how to train the recurrent network. If batch_trajectory is True, the data
            will like (batch_size, trajectory_length, feature_dim), you can see as (batch_size,time_step,feature_dim).
            If batch_trajectory is False, the data will like (batch_size, 1, feature_dim).
            action_space_type (str): action space type. It can be 'discrete' or 'continuous'. Default is None.
            None means the action space type is continuous.
        """
        super().__init__()
        self.epoch_length = epoch_length
        self.n_epochs = n_epochs
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.min_buffer_size = 0
        self.display_interval = 1
        self.calculate_episodes = 5
        self.update_times = update_times
        self.batch_trajectory = batch_trajectory
        self.store_model_times = store_model_times

        self.adjust_parameters = []

        self.action_space_type = action_space_type

        self.eval_episodes = eval_episodes
        self.eval_interval = eval_interval

    def point_adjust_parameter(self, names: list):
        """
        For higher level algorithm, you can adjust the parameter in this function.
        """

        self.adjust_parameters = names

        # check the parameter name whether is in the class
        for name in names:
            if not hasattr(self, name):
                raise AttributeError(f'{name} is not in {self.__class__.__name__} class')

    def set_adjust_parameter_value(self, parm_dict: dict):
        for key, value in parm_dict.items():
            setattr(self, key, value)


class SAC_parameter(BaseParameter):

    def __init__(self, epoch_length: int, n_epochs: int, batch_size: int,
                 update_interval: int,
                 discount: float, tau: float,
                 store_model_times=5,
                 eval_episodes=0,
                 action_space_type: str = None,
                 eval_interval: int = 1,
                 ):
        """
        Parameters of SAC algorithm.

        Reference:
        ----------
        [1] Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum
        entropy deep reinforcement learning with a stochastic actor."
        arXiv preprint arXiv:1801.01290 (2018).

        Args:
            epoch_length (int): length of epoch.
            n_epochs (int): times of optimizing the network.
            batch_size (int): batch size.
            update_interval (int): update interval.
            discount (float): discount factor.
            tau (float): Soft value function target update weight.
            store_model_times (int): times of storing the model.
            action_space_type (str): action space type. It can be 'discrete' or 'continuous'. Default is None.
            None means the action space type is continuous.


        """
        super().__init__(epoch_length, n_epochs, batch_size, update_interval,
                         store_model_times=store_model_times,
                         action_space_type=action_space_type,
                         eval_episodes=eval_episodes,
                         eval_interval=eval_interval,
                         )

        self.discount = discount
        self.tau = tau
        # self.learning_rate = learning_rate
        # self.update_interval = update_interval


class SAC2_parameter(BaseParameter):

    def __init__(self, episode_length: int, n_epochs: int,
                 batch_size: int,
                 discount: float,
                 tau: float,
                 buffer_size: int,
                 mini_buffer_size: int,
                 alpha_learning_rate: float = 3e-4,
                 update_times: int = 1,
                 calculate_episodes: int = 5,
                 display_interval: int = 1,
                 update_interval: int = 0,
                 store_model_times=5,
                 action_space_type: str = None,
                 eval_episodes=0,
                 eval_interval: int = 1,
                 ):
        """
        Parameters of SAC2 algorithm.

        Example:
        --------
        >>> from AquaML.rlalgo.Parameters import SAC2_parameter
        >>> sac_parameter = SAC2_parameter(
        ...     episode_length=200,
        ...     n_epochs=1000,
        ...     batch_size=128,
        ...     discount=0.99,
        ...     tau=0.005,
        ...     buffer_size=100000,
        ...     mini_buffer_size=1000,
        ...     update_interval=1000,
        ...     display_interval=1,
        ...     calculate_episodes=5,
        ...     update_times=1,
        ...     alpha_learning_rate=3e-4
        ... )

        This means that the algorithm will run 1000 epochs, each epoch has 1000 (update_interval) steps.
        The update information will be displayed every 1 epoch.
        The algorithm will calculate the average reward of 5 episodes in each thread every 1 epoch.
        The algorithm will update the network 1 (update_times) times every 1000 steps.
        The algorithm will pre sample 1000 (mini_buffer_size) samples from the buffer every 1000 steps.
        The algorithm will use 100000 (buffer_size) samples to store the experience.
        the algorithm will use 0.99 (discount) as the discount factor.
        The algorithm will use 0.005 (tau) as the target network update weight.
         The algorithm will use 3e-4 (alpha_learning_rate) as the alpha learning rate.

        Reference:
        ----------
        [1] Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications."
        arXiv preprint arXiv:1812.05905 (2018).

        Args:
            episode_length (int): length of episode.
            n_epochs (int): times of optimizing the network.
            batch_size (int): batch size.
            discount (float): discount factor.
            tau (float): Soft q function target update weight.
            buffer_size (int): buffer size.
            mini_buffer_size (int): mini buffer size. Algo will pre-sample mini buffer from buffer.
            alpha_learning_rate (float, optional): learning rate of alpha. Defaults to 3e-4.
            update_times (int): update times for each epoch.
            update_interval (int, optional): update interval. Defaults to 0.
            calculate_episodes (int, optional): calculate episodes. Defaults to 5.
            display_interval (int, optional): display interval which depends on epoch. Defaults to 1.
        """
        super().__init__(episode_length,
                         n_epochs,
                         buffer_size,
                         batch_size,
                         update_interval,
                         store_model_times=store_model_times,
                         action_space_type=action_space_type,
                         eval_episodes=eval_episodes,
                         eval_interval=eval_interval,
                         )

        self.discount = discount
        self.tau = tau
        self.update_interval = update_interval
        self.alpha_learning_rate = alpha_learning_rate
        self.mini_buffer_size = mini_buffer_size

        self.gamma = discount

        self.display_interval = display_interval

        self.calculate_episodes = calculate_episodes

        self.update_times = update_times


class PPO_parameter(BaseParameter):

    def __init__(self,
                 epoch_length: int,
                 n_epochs: int,
                 total_steps: int,
                 batch_size: int,
                 update_times: int = 4,
                 update_critic_times: int = 1,
                 update_actor_times: int = 1,
                 entropy_coeff: float = 0.01,
                 epsilon: float = 0.2,
                 gamma: float = 0.99,
                 lambada: float = 0.95,
                 calculate_episodes: int = 5,
                 store_model_times=5,
                 action_space_type: str = None,
                 batch_advantage_normlization: bool = False,
                 batch_trajectory: bool = False,
                 eval_episodes=0,
                 eval_interval: int = 1,
                 ):
        """
        Parameters of PPO algorithm.

        Reference:
        ----------
        [1] Schulman, John, et al. "Proximal policy optimization algorithms."
        arXiv preprint arXiv:1707.06347 (2017).

        Args:
            epoch_length (int): length of epoch.
            total_steps (int): total steps for one epoch. Also, can be seen as the buffer size.
            n_epochs (int): times of optimizing the network.
            batch_size (int): batch size.
            epsilon (float): epsilon for clipping. Also, can be seen as the clip range.
            gamma (float): discount factor.
            lambada (float): lambada for GAE.
            entropy_coeff (float): entropy coefficient.
            update_times (int): update times for each epoch.
            update_critic_times (int): update critic times for each epoch.
            update_actor_times (int): update actor times for each epoch.
            batch_advantage_normlization (bool, optional): whether to normalize the advantage. Defaults to False.


        """
        super().__init__(epoch_length,
                         n_epochs,
                         total_steps,
                         batch_size,
                         update_times=update_times,
                         store_model_times=store_model_times,
                         action_space_type=action_space_type,
                         eval_episodes=eval_episodes,
                         eval_interval=eval_interval,
                         )

        self.gamma = gamma
        self.lambada = lambada
        self.entropy_coeff = entropy_coeff
        self.epsilon = epsilon
        self.update_critic_times = update_critic_times
        self.update_actor_times = update_actor_times
        self.calculate_episodes = calculate_episodes
        self.summary_episodes = self.calculate_episodes
        self.batch_trajectory = batch_trajectory
        self.batch_advantage_normlization = batch_advantage_normlization
        # self.learning_rate = learning_rate
        # self.update_interval = update_interval


class FusionPPO_parameter(BaseParameter):

    def __init__(self,
                 epoch_length: int,
                 n_epochs: int,
                 total_steps: int,
                 batch_size: int,
                 update_times: int = 4,
                 update_critic_times: int = 1,
                 update_actor_times: int = 1,
                 entropy_coeff: float = 0.01,
                 batch_trajectory: bool = False,
                 epsilon: float = 0.2,
                 gamma: float = 0.99,
                 lambada: float = 0.95,
                 batch_advantage_normalization: bool = False,
                 action_space_type: str = None,
                 eval_episodes=0,
                 eval_interval: int = 1,
                 ):
        """
        Parameters of Fusion PPO algorithm.

        Args:
            epoch_length (int): length of epoch.
            total_steps (int): total steps for one epoch. Also, can be seen as the buffer size.
            n_epochs (int): times of optimizing the network.
            batch_size (int): batch size.
            epsilon (float): epsilon for clipping. Also, can be seen as the clip range.
            gamma (float): discount factor.
            lambada (float): lambada for GAE.
            entropy_coeff (float): entropy coefficient.
        """
        super().__init__(epoch_length,
                         n_epochs,
                         total_steps,
                         batch_size,
                         update_times=update_times,
                         batch_trajectory=batch_trajectory,
                         action_space_type=action_space_type,
                         eval_episodes=eval_episodes,
                         eval_interval=eval_interval,
                         )

        self.gamma = gamma
        self.lambada = lambada
        self.entropy_coeff = entropy_coeff
        self.epsilon = epsilon
        self.update_critic_times = update_critic_times
        self.update_actor_times = update_actor_times
        self.batch_advantage_normalization = batch_advantage_normalization

        self.calculate_episodes = int(self.buffer_size / self.epoch_length)
        self.summary_episodes = self.calculate_episodes


class BehaviorCloning_parameter(BaseParameter):
    def __init__(self,
                 epoch_length: int,
                 n_epochs: int,
                 buffer_size: int,
                 batch_size: int,
                 ):
        super().__init__(epoch_length,
                         n_epochs,
                         buffer_size,
                         batch_size, )
