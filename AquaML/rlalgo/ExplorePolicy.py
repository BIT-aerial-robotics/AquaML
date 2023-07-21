import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import abc

pi = tf.constant(np.pi)
e = tf.constant(np.e)


def create_explor_policy(explore_policy_name, shape, actor_out_names):
    if explore_policy_name == 'Gaussian':
        policy = GaussianExplorePolicy(shape)
    elif explore_policy_name == 'OrnsteinUhlenbeck':
        policy = OrnsteinUhlenbeckExplorePolicy(shape)
    elif explore_policy_name == 'NoExplore':
        policy = VoidExplorePolicy(shape)
    elif explore_policy_name == 'Categorical':
        policy = CategoricalExplorePolicy(shape)
    elif explore_policy_name == 'Void':
        policy = VoidExplorePolicy(shape)
    else:
        raise NotImplementedError(f'{explore_policy_name} is not implemented.')

    create_info = policy.create_info()

    infos = []

    for name, info in create_info.items():
        if name in actor_out_names:
            pass
        else:
            infos.append(info)

    return policy, infos


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.x_prev = None
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


# TODO: 探索策略的创建需要优化
class ExplorePolicyBase(abc.ABC):
    """Explore policy base class.
    
    Example of using this class:
    explore_policy = GaussianExplorePolicy(shape)
    action, prob = explore_policy(mu, log_std)
    
    New explore policy should inherit this class.
    Another things you should do:
    1. declare the input_info. It is a tuple. It is used to get the input of explore policy.
    Notice: the order of input_info is the same as the order of input of scale_out function.
    

    """

    def __init__(self, shape):
        self.shape = shape
        self.input_name = None  # tuple

        self._aditional_output = {}

    @abc.abstractmethod
    def noise_and_prob(self, batch_size=1):
        """
        This function must use tf.function to accelerate.
        All the exploration noise use reparameter tricks.
        """

    @abc.abstractmethod
    def scale_out(self, *args, **kwargs):
        """
        Scale the action to the range of environment.
        """

    @abc.abstractmethod
    def test_action(self, *args, **kwargs):
        """
        Test action.
        """

    def __call__(self, inputs_dict: dict, test_flag=False):
        """inputs_dict is a dict. The key is the name of input. The value is the input.
        inputs_dict must contain all the output of actor model.
        such as:
        inputs_dict = {'mu':mu, 'log_std':log_std}
        
        args:
        inputs_dict (dict): dict of inputs. The key is the name of input. The value is the input.
        return:
        action (tf.Tensor): action of environment.
        prob (tf.Tensor): probability of action.
        """
        # get inputs from inputs_dict

        inputs = []

        for key in self.input_name:
            inputs.append(inputs_dict[key])

        if test_flag:
            return self.test_action(*inputs)
        else:
            return self.scale_out(*inputs)

    @abc.abstractmethod
    def resample_prob(self, mu, log_std, action):
        """
        Resample the probability of action.
        """

    def reset(self):
        """
        如果需要重置，需要重写这个函数
        """
        pass

    def get_entropy(self, *args, **kwargs):
        """
        获取策略的熵
        """
        pass

    def create_info(self):
        """
        创建探索策略需要的额外输入，比如高斯策略，sigma的输入
        """
        dic = {}

        return dic

    @property
    def get_aditional_output(self):
        return self._aditional_output


class GaussianExplorePolicy(ExplorePolicyBase):
    def __init__(self, shape):
        super().__init__(shape)
        mu = tf.zeros(shape, dtype=tf.float32)
        sigma = tf.ones(shape, dtype=tf.float32)
        self.dist = tfp.distributions.Normal(loc=mu, scale=sigma)
        self.input_name = ('action', 'log_std')

        self._aditional_output = {
            'prob': {
                'shape': self.shape,
                'dtype': np.float32,
            }
        }

    @tf.function
    def noise_and_prob(self, batch_size=1):
        noise = self.dist.sample(batch_size)
        prob = self.dist.prob(noise)

        return noise, prob

    def get_prob(self, action):
        prob = self.dist.prob(action)
        return prob

    def scale_out(self, mu, log_std):
        sigma = tf.exp(log_std)
        batch_size = mu.shape[0]
        noise, prob = self.noise_and_prob(batch_size)
        action = mu + sigma * noise

        # action = tf.clip_by_value(action, -1, 1)

        # noise = (action - mu) / sigma
        #
        # prob = self.get_prob(noise)

        return action, prob

    def resample_prob(self, mu, std, action):
        # sigma = tf.exp(log_std)
        noise = (action - mu) / std
        log_prob = self.dist.log_prob(noise)

        # dist = tfp.distributions.Normal(loc=mu, scale=std ** 2)
        # log_prob = dist.log_prob(action)

        return log_prob

    def get_entropy(self, mean, log_std):

        dist = tfp.distributions.Normal(loc=mean, scale=tf.exp(log_std))

        entropy = tf.reduce_sum(dist.entropy(), axis=1, keepdims=True)
        # if len(mean.shape) == 1:
        #     entropy = tf.reduce_sum(dist.entropy())
        # else:
        #     entropy = tf.reduce_sum(dist.entropy(), axis=1)

        return entropy

    def test_action(self, mu, log_std):
        return mu, tf.ones((1, *self.shape))

    def create_info(self):
        log_std = {
            'name': 'log_std',
            'shape': self.shape,
            'trainable': True,
            'dtype': np.float32,
        }

        dict_info = {
            'log_std': log_std,
        }
        return dict_info


class VoidExplorePolicy(ExplorePolicyBase):
    def __init__(self, shape):
        super().__init__(shape)
        self.input_name = ('action',)
        self._aditional_output = {
            'prob': {
                'shape': self.shape,
                'dtype': np.float32,
            }
        }

    @tf.function
    def noise_and_prob(self, batch_size=1):
        return tf.zeros((batch_size, *self.shape)), tf.ones((batch_size, *self.shape))

    def scale_out(self, mu):
        return mu, tf.ones((1, *self.shape))

    def resample_prob(self, mu, log_std, action):
        return tf.ones((1, *self.shape))

    def test_action(self, mu):
        return mu, tf.ones((1, *self.shape))

    def get_entropy(self, mean, log_std):
        return 1


# 离散探索策略
class CategoricalExplorePolicy(ExplorePolicyBase):
    def __init__(self, shape):
        super().__init__(shape)
        self.input_name = ('action',)

    # @tf.function
    def noise_and_prob(self, batch_size=1):
        pass

    def scale_out(self, action):
        dist = tfp.distributions.Categorical(logits=action)
        sample_action = dist.sample()
        prob = dist.prob(sample_action)

        return sample_action, prob

    def resample_prob(self, log_prob, action):
        dist = tfp.distributions.Categorical(logits=log_prob)
        log_prob = dist.log_prob(action)
        return log_prob

    def test_action(self, action):
        action = tf.argmax(action, axis=-1)

        return action, tf.ones((1, *self.shape))


class OrnsteinUhlenbeckExplorePolicy(ExplorePolicyBase):
    """
    该探索策略用于确定性策略。
    """

    def __init__(self, shape, mu=0, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        super().__init__(shape)
        self.input_name = ('action',)
        self.noise = OrnsteinUhlenbeckActionNoise(mu, sigma, theta, dt, x0)

    def noise_and_prob(self, batch_size=1):
        pass

    def scale_out(self, action):
        noise = self.noise()
        action = action + noise
        return action, tf.ones((1, *self.shape))

    def resample_prob(self, log_prob, action):
        pass

    def test_action(self, action):
        return action, tf.ones((1, *self.shape))

    def reset(self):
        self.noise.reset()
