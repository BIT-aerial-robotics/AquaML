import abc

from AquaML.core.BaseParameter import BaseParameter


class BaseAgentParameter(BaseParameter, abc.ABC):
    def __init__(self,
                 rollout_steps: int,
                 max_steps: int,
                 epochs: int,
                 batch_size: int,
                 update_times: int,
                 eval_interval: int,
                 eval_episodes: int,
                 eval_episode_length: int,
                 explore_policy='Default',

                 ):
        """
        2.1版本中，优化参数设置，更加人性化，更加易于理解。
        2.1将全面支持不定长MDP。

        Args:
            rollout_steps (int): 单个线程的rollout步数。
            max_steps (int): 单个回合最大步数。如果不想使用此限制，直接设置很大的数值即可。
            epochs (int): 算法的训练轮数。
            batch_size (int): 算法的训练批次大小。
            update_times (int): 每一个epoch模型更新次数。
            eval_interval (int): 评估间隔。
            eval_episodes (int): 评估回合数。
            eval_episode_length (int): 评估回合长度。
            explore_policy (str, optional): 探索策略。默认为'Default'，即使用算法自带的探索策略。
                                            如果想使用自定义的探索策略，需要在此处传入探索策略的类名。
        """
        super().__init__()
        self.rollout_steps = rollout_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.explore_policy = explore_policy

        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.eval_episode_length = eval_episode_length

        self.max_steps = max_steps

        self.update_times = update_times

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


class DDPGAgentParameter(BaseAgentParameter):

    def __init__(self,
                 rollout_steps: int,
                 epochs: int,
                 batch_size: int = 128,
                 tau: float = 0.001,
                 eval_interval: int = 1000,
                 explore_policy='Default',
                 ):
        super().__init__(
            rollout_steps=rollout_steps,
            epochs=epochs,
            batch_size=batch_size,
            explore_policy=explore_policy,
        )

        self.tau = tau


class PPOAgentParameter(BaseAgentParameter):

    def __init__(self,
                 rollout_steps: int,
                 epochs: int,
                 batch_size: int,
                 update_times: int,
                 update_critic_times: int,
                 update_actor_times: int,
                 batch_advantage_normalization: bool = True,
                 clip_ratio: float = 0.2,
                 entropy_coef: float = 0.01,
                 gamma: float = 0.99,
                 lamda: float = 0.95,
                 eval_episode_length: int = 1000,
                 eval_interval: int = 10,
                 eval_episodes: int = 1,
                 explore_policy='Default',
                 max_steps: int = 1000000,
                 log_std_init_value: float = -0.5,
                 ):
        """
            Initializes the parameters for the PPO agent.

            Args:
            - rollout_steps (int): The number of steps to rollout the environment.
            - epochs (int): The number of epochs to train the agent.
            - batch_size (int): The size of the batch to use for training.
            - update_times (int): The number of times to update the agent.
            - update_critit_times (int): The number of times to update the critic.
            - update_actor_times (int): The number of times to update the actor.
            - gamma (float): The discount factor for rewards.
            - lamda (float): The lambda value for GAE.
            - eval_episode_length (int): The maximum length of an evaluation episode.
            - eval_interval (int): The interval at which to evaluate the agent.
            - eval_episodes (int): The number of episodes to evaluate the agent.
            - explore_policy (str): The exploration policy to use.
            - max_steps (int): The maximum number of steps to take in the environment.
            - log_std_init_value (float): The initial value for the log standard deviation.
            """
        super().__init__(
            rollout_steps=rollout_steps,
            epochs=epochs,
            batch_size=batch_size,
            explore_policy=explore_policy,
            eval_episode_length=eval_episode_length,
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
            max_steps=max_steps,
            update_times=update_times,
        )

        self.log_std_init_value = log_std_init_value

        self.gamma = gamma
        self.lamda = lamda

        self.update_times = update_times
        self.update_critic_times = update_critic_times
        self.update_actor_times = update_actor_times

        self.clip_ratio = clip_ratio

        self.entropy_coef = entropy_coef

        self.batch_advantage_normalization = batch_advantage_normalization