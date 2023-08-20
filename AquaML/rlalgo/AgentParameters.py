import abc

from AquaML.core.BaseParameter import BaseParameter


class BaseAgentParameter(BaseParameter, abc.ABC):
    def __init__(self,
                 rollout_steps: int,
                 max_steps: int,
                 min_steps: int,
                 epochs: int,
                 batch_size: int,
                 update_times: int,
                 eval_interval: int,
                 eval_episodes: int,
                 eval_episode_length: int,
                 checkpoint_interval: int,

                 summary_style:str = 'episode',
                 summary_steps:int = 1,
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
            checkpoint_interval (int): 模型保存间隔.
            summary_style (str): summary的类型，可选episode和step，默认为episode。当为episode时，每个episode结束后记录一次summary，当为step时，每多少个斯特普求和。
            summary_steps (int): summary的间隔，当summary_style为step时，每多少个step记录一次summary。
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

        self.checkpoint_interval = checkpoint_interval

        self.min_steps = min_steps

        self.summary_style = summary_style
        self.summary_steps = summary_steps

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
                 update_critic_times=1,
                 update_actor_times=1,
                 batch_advantage_normalization: bool = True,
                 target_kl: float = None,
                 train_all: bool = False,
                 train_fusion: bool = False,
                 vf_coef: float = 0.5,
                 min_steps: int = 1,
                 checkpoint_interval: int = 10,
                 clip_ratio: float = 0.2,
                 entropy_coef: float = 0.01,
                 gamma: float = 0.99,
                 lamda: float = 0.95,
                 eval_episode_length: int = 1000,
                 eval_interval: int = 10,
                 eval_episodes: int = 1,
                 explore_policy='Default',
                 max_steps: int = 1000000,
                 log_std_init_value: float = -0.0,

                 # summary style
                 summary_style:str = 'episode',
                 summary_steps:int = 1,

                 # sequential training
                 is_sequential: bool = False,
                 sequential_args: dict = {
                     'split_point_num': 1,
                     'return_first_hidden': True,
                 },
                 shuffle: bool = False,
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
            - is_sequential (bool): Whether to use sequential training.
            - sequential_args (dict): The arguments for sequential training.
            - shuffle (bool): Whether to shuffle the data according to the environment. We suggest setting this to True, when using sequential training.
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
            checkpoint_interval=checkpoint_interval,
            min_steps=min_steps,
            summary_style=summary_style,
            summary_steps=summary_steps,
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

        self.vf_coef = vf_coef

        self.target_kl = target_kl

        self.train_all = train_all

        self.is_sequential = is_sequential
        self.sequential_args = sequential_args
        self.shuffle = shuffle

        self.train_fusion = train_fusion
        
    
class AMPAgentParameter(BaseAgentParameter):
    """
    
    Adversarial motion prior agent parameter
    
    """
    
    def __init__(self,
                 rollout_steps: int,
                 epochs: int,
                 batch_size: int,
                 k_batch_size: int,
                 update_times: int,
                 update_critic_times=1,
                 update_actor_times=1,
                 update_discriminator_times: int = 4096,
                 discriminator_replay_buffer_size: int = 1e5,
                 batch_advantage_normalization: bool = True,
                 target_kl: float = None,
                 train_all: bool = False,
                 train_fusion: bool = False,
                 vf_coef: float = 0.5,
                 gp_coef: float = 10.0,
                 task_rew_coef: float = 0.5,
                 style_rew_coef: float = 0.5,
                 min_steps: int = 1,
                 checkpoint_interval: int = 10,
                 clip_ratio: float = 0.2,
                 entropy_coef: float = 0.01,
                 gamma: float = 0.99,
                 lamda: float = 0.95,
                 eval_episode_length: int = 1000,
                 eval_interval: int = 10,
                 eval_episodes: int = 1,
                 explore_policy='Default',
                 max_steps: int = 1000000,
                 log_std_init_value: float = -0.0,
                 

                 # summary style
                 summary_style:str = 'episode',
                 summary_steps:int = 1,

                 # sequential training
                 is_sequential: bool = False,
                 sequential_args: dict = {
                     'split_point_num': 1,
                     'return_first_hidden': True,
                 },
                 shuffle: bool = False,
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
            - is_sequential (bool): Whether to use sequential training.
            - sequential_args (dict): The arguments for sequential training.
            - shuffle (bool): Whether to shuffle the data according to the environment. We suggest setting this to True, when using sequential training.
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
            checkpoint_interval=checkpoint_interval,
            min_steps=min_steps,
            summary_style=summary_style,
            summary_steps=summary_steps,
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

        self.vf_coef = vf_coef

        self.target_kl = target_kl

        self.train_all = train_all

        self.is_sequential = is_sequential
        self.sequential_args = sequential_args
        self.shuffle = shuffle

        self.train_fusion = train_fusion
        
        # AMP specific parameters
        self.update_discriminator_times = update_discriminator_times
        self.k_batch_size = k_batch_size
        self.gp_coef = gp_coef
        self.discriminator_replay_buffer_size = discriminator_replay_buffer_size
        
        self.task_rew_coef = task_rew_coef
        self.style_rew_coef = style_rew_coef


if __name__ == '__main__':
    
    ppo_parameter = PPOAgentParameter(
        rollout_steps=10,
        epochs=10,
        batch_size=10,
        update_times=10,
        update_critic_times=10,
        update_actor_times=10,
        gamma=0.99,
        lamda=0.95,
        eval_episode_length=1000,
        eval_interval=10,
        eval_episodes=1,
        explore_policy='Default',
        max_steps=1000000,
        log_std_init_value=-0.0,
        is_sequential=False,
        sequential_args={
            'split_point_num': 1,
            'return_first_hidden': True,
        },
        shuffle=False,
    )
    
    amp_parameter = AMPAgentParameter(
        PPO_parameter=ppo_parameter,
    )
    
    print(amp_parameter.batch_size)