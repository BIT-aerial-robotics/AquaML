import tensorflow as tf
import tensorflow_probability as tfp
import abc

class ExplorePolicyBase(abc.ABC):

    def __init__(self, shape):
        self.shape = shape
    
    @abc.abstractmethod
    def noise_and_prob(self):
        """
        This function must use tf.function to accelerate.
        All the exploration noise use reparameter tricks.
        """


class GuassianExplorePolicy(ExplorePolicyBase):
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