import tensorflow as tf
import tensorflow_probability as tfp
import abc


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


class GaussianExplorePolicy(ExplorePolicyBase):
    def __init__(self, shape):
        super().__init__(shape)
        mu = tf.zeros(shape, dtype=tf.float32)
        sigma = tf.ones(shape, dtype=tf.float32)
        self.dist = tfp.distributions.Normal(loc=mu, scale=sigma)
        self.input_name = ('action', 'log_std')

    @tf.function
    def noise_and_prob(self, batch_size=1):
        noise = self.dist.sample(batch_size)
        prob = self.dist.prob(noise)

        return noise, prob

    def scale_out(self, mu, log_std):
        sigma = tf.exp(log_std)
        noise, prob = self.noise_and_prob()
        action = mu + sigma * noise

        return action, prob

    def resample_prob(self, mu, std, action):
        # sigma = tf.exp(log_std)
        dist = tfp.distributions.Normal(loc=mu, scale=std)
        log_prob = dist.log_prob(action)
        return log_prob

    def test_action(self, mu, log_std):
        return mu, tf.ones((1, *self.shape))


class VoidExplorePolicy(ExplorePolicyBase):
    def __init__(self, shape):
        super().__init__(shape)
        self.input_name = ('action',)

    @tf.function
    def noise_and_prob(self, batch_size=1):
        return tf.zeros((batch_size, *self.shape)), tf.ones((batch_size, *self.shape))

    def scale_out(self, mu):
        return mu, tf.ones((1, *self.shape))

    def resample_prob(self, mu, log_std, action):
        return tf.ones((1, *self.shape))

    def test_action(self, mu):
        return mu, tf.ones((1, *self.shape))


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
