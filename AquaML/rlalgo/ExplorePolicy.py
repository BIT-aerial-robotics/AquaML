import tensorflow as tf
import tensorflow_probability as tfp
import abc

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
        self.input_info = None # tuple
    
    @abc.abstractmethod
    def noise_and_prob(self):
        """
        This function must use tf.function to accelerate.
        All the exploration noise use reparameter tricks.
        """
    @abc.abstractmethod
    def scale_out(self, *args, **kwargs):
        """
        Scale the action to the range of environment.
        """
    def __call__(self, inputs_dict:dict):
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
        
        for key in self.input_info:
            inputs.append(inputs_dict[key])
        
        return self.scale_out(*inputs)


class GaussianExplorePolicy(ExplorePolicyBase):
    def __init__(self, shape):
        super().__init__(shape)
        mu = tf.zeros(shape, dtype=tf.float32)
        sigma = tf.ones(shape, dtype=tf.float32)
        self.dist = tfp.distributions.Normal(loc=mu, scale=sigma)
    
    @tf.function
    def noise_and_prob(self):
        noise = self.dist.sample()
        prob = self.dist.prob(noise)
        
        return noise, prob
    
    def scale_out(self, mu, log_std):
        sigma = tf.exp(log_std)
        noise, prob = self.noise_and_prob()
        action = mu + sigma * noise
        
        return action, prob