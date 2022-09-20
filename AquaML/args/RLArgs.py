# from AquaML.args.TrainArgs import TrainArgs
from AquaML.data.ArgsUnit import ArgsUnit


class TrainArgs:
    def __init__(self,
                 # lstm训练参数
                 burn_in: int or None = None,
                 traj_length: int or None = None,
                 overlap_size: int or None = None,
                 actor_is_batch_timesteps=False,
                 critic_is_batch_timesteps=False
                 ):
        """
        This args points to each algorithm, it controls pre-process of data.

        :param burn_in: (int) Use some data to initiate networks hidden state.
        """
        self.burn_in = burn_in
        self.traj_length = traj_length
        self.overlap_size = overlap_size
        self.actor_is_batch_timesteps = actor_is_batch_timesteps
        self.critic_is_batch_timesteps = critic_is_batch_timesteps
        self.args = {'burn_in': self.burn_in, 'traj_length': self.traj_length,
                     'overlap_size': self.overlap_size, 'actor_is_batch_timesteps': self.actor_is_batch_timesteps,
                     'critic_is_batch_timesteps': self.critic_is_batch_timesteps}


class EnvArgs:
    def __init__(self, max_steps, total_steps, worker_num=1, episode_clip=None):
        self.max_steps = max_steps
        self.total_steps = total_steps
        self.worker_num = worker_num

        self.total_steps = total_steps

        # computing each thread
        self.one_thread_total_steps = int(self.total_steps / worker_num)
        self.episode_clip = episode_clip
        if self.episode_clip is None:
            self.episode_clip = max_steps

        self.args = {
            'max_steps': self.max_steps,
            'total_steps': total_steps,
            'workers': self.worker_num,
            'one thread total steps': self.one_thread_total_steps,
            'episode clip': self.episode_clip
        }

    def sync(self, process_id):
        """
        Provide task for every thread.

        :param process_id:
        :return:
        """
        start_point = self.one_thread_total_steps * (process_id - 1)
        end_pointer = self.one_thread_total_steps * process_id - 1

        return start_point, end_pointer


class PPOHyperParam:
    def __init__(self, epochs=100, clip_ratio=0.1, actor_learning_rate=2e-4,
                 critic_learning_rate=2e-3, entropy_ratio=0.00, gamma=0.99,
                 lambada=0.95, update_critic_times=1, update_actor_times=1,
                 update_times=1, batch_size=64):
        """

        :param epochs:
        :param clip_ratio:
        :param actor_learning_rate:
        :param critic_learning_rate:
        :param entropy_ratio:
        :param gamma:
        :param lambada:
        :param update_critic_times:
        :param update_actor_times:
        :param update_times:
        :param batch_size:
        """
        self.clip_ratio = clip_ratio
        self.entropy_ratio = entropy_ratio
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.update_times = update_times
        self.batch_size = batch_size
        self.epochs = epochs
        self.gamma = gamma
        self.lambada = lambada

        self.update_actor_times = update_actor_times
        self.update_critic_times = update_critic_times

        self.args = {
            'clip ratio': self.clip_ratio,
            'entropy ratio': self.entropy_ratio,
            'actor learning rate': self.actor_learning_rate,
            'critic learning rate': self.critic_learning_rate,
            'update times': self.update_times,
            'batch size': self.batch_size,
            'epochs': self.epochs,
            'gamma': self.gamma,
            'lambada': lambada,
            'update actor times': update_actor_times,
            'update critic times': update_critic_times
        }


class PPGHyperParam:
    def __init__(self, epochs=100, clip_ratio=0.1, actor_learning_rate=2e-3,
                 critic_learning_rate=2e-3, entropy_ratio=0.00, gamma=0.99,
                 lambada=0.95, update_critic_times=1, update_actor_times=1,
                 c1=0.5, update_times=1, batch_size=64, recovery_epoch=-1, recovery_update_steps=8, c2=1):
        """

        :param epochs:
        :param clip_ratio:
        :param actor_learning_rate:
        :param critic_learning_rate:
        :param entropy_ratio:
        :param gamma:
        :param lambada:
        :param update_critic_times:
        :param update_actor_times:
        :param update_times:
        :param batch_size:
        :param recovery_epoch: pre-train the critic network, -1 means not pre-train.
        """
        self.clip_ratio = clip_ratio
        self.entropy_ratio = entropy_ratio
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.update_times = update_times
        self.batch_size = batch_size
        self.epochs = epochs
        self.gamma = gamma
        self.lambada = lambada

        self.update_actor_times = update_actor_times
        self.update_critic_times = update_critic_times

        self.c1 = c1

        self.recovery_epoch = recovery_epoch
        self.recovery_update_steps = recovery_update_steps

        self.c2 = c2

        self.args = {
            'clip ratio': self.clip_ratio,
            'entropy ratio': self.entropy_ratio,
            'actor learning rate': self.actor_learning_rate,
            'critic learning rate': self.critic_learning_rate,
            'update times': self.update_times,
            'batch size': self.batch_size,
            'epochs': self.epochs,
            'gamma': self.gamma,
            'lambada': lambada,
            'update actor times': update_actor_times,
            'update critic times': update_critic_times,
            'c1(entropy ratio)': c1,
            "c2": c2,
            "recovery epoch": recovery_epoch,
            "recovery update steps":recovery_update_steps
        }


class TaskArgs:
    def __init__(self, algo_param, obs_info: dict, actor_inputs_info: list, actor_outputs_info: dict,
                 critic_inputs_info: list,
                 reward_info: list, distribution_info: dict, env_args: EnvArgs, training_args: TrainArgs):
        """
        Config RL task. Reward_info must contain key 'total_reward'. If share parameter actor critic is used, the
        actor_inputs_info is equal to critic_inputs_info but the actor_outputs_info likes {'action','value',...}.
        If the policy is stochastic, the actor_outputs_info likes {'action', 'prob', ...}. So if share parameter
        and stochastic are usd for the policy, the actor_outputs_info likes {'action', 'prob', 'value'}.
        The distribution_info contains key "is_distribution".

        :param reward_info: The information of reward.
        :param obs_info: (dict) The output information of env.
        :param actor_inputs_info: (dict) The inputs information of actor.
        :param actor_outputs_info: (dict) The outputs information of actor.
        :param critic_inputs_info: (dict) The input information of critic.
        :param distribution_info: (dict) The information of distribution.
        :param env_args: (EnvArgs) Roll information of the environment.
        """

        self.obs_info = obs_info
        self.actor_inputs_info = actor_inputs_info
        self.critic_inputs_info = critic_inputs_info

        # check the dict
        if actor_outputs_info.get('action') is None:
            raise ValueError('actor_outputs_info must have key "action".')

        self.actor_outputs_info = actor_outputs_info

        # if reward_info.get('total_reward') is None:
        #     raise ValueError('reward_info must have key "total reward"')

        self.reward_info = reward_info

        if distribution_info.get('is_distribution') is None:
            raise ValueError('distribution_info must have the key "is_distribution".')

        self.distribution_info = distribution_info

        self.env_args = env_args

        # self.total_length = total_length

        self.algo_param = algo_param

        self.training_args = training_args
