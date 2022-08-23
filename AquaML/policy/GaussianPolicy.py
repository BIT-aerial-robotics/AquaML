import tensorflow as tf
import tensorflow_probability as tfp
from AquaML.policy.BasePolicy import BasePolicy
from AquaML.data.ParamData import ParamData
import AquaML as A


class GaussianPolicy(BasePolicy):
    def __init__(self, model: tf.keras.Model, name: str):
        super().__init__(model=model, name=name)
        self.type = A.STOCHASTIC
        self.distribution = None
        self.hierarchical = None
        self.tf_log_std = None
        self.log_std = None
        # self.model_type = model_type

    def create_log_std(self, shape, hierarchical, work_space: str):
        if hierarchical == 0 or hierarchical == 1:
            self.log_std = ParamData(work_space + '_' + self.name + '_log_std', shape=shape, hierarchical=hierarchical)
            self.tf_log_std = tf.Variable(self.log_std.data - 0.5, trainable=True)
            self.hierarchical = hierarchical
        else:
            self.log_std = ParamData(work_space + '_' + self.name + '_log_std', shape=shape, hierarchical=hierarchical)
            self.tf_log_std = tf.cast(self.log_std.data, dtype=tf.float32)
            self.hierarchical = hierarchical

    def create_distribution(self, shape):
        mu = tf.zeros(shape=shape, dtype=tf.float32)
        sigma = tf.ones(shape=shape, dtype=tf.float32)
        self.distribution = tfp.distributions.Normal(mu, sigma)

    # @tf.function
    # def noise_and_prob(self):
    #     noise = self.distribution.sample()
    #     prob = self.distribution.prob(noise)
    #
    #     prob = tf.clip_by_value(prob, 1e-6, 1)
    #     prob = tf.squeeze(prob)
    #
    #     return noise, prob

    def get_action(self, *args):
        """
        Used for interacting with environment.

        :param args: (obs1,obs2, ... , training)
        :return: (action, prob, value, hidden_state ,other information)
        """
        # print(*args)
        if self.log_std:
            out = self.run_model(*args)
            if isinstance(out, tuple):
                mu = out[0]
            else:
                mu = out
            sigma = tf.exp(self.tf_log_std)
        else:
            out = self.run_model(*args)
            mu = out[0]
            sigma = tf.exp(out[1])

        noise, prob = self.noise_and_prob()

        action = mu + noise * sigma

        if isinstance(out, tuple):
            out = (action, prob, *out[2:])
        else:
            out = (action, prob)

        return out

    # def set_log_std(self, value):
    #     self.log_std.set_log_std(value)

    def sync(self):
        if self.hierarchical == 0 or self.hierarchical == 1:
            print("save_complete")
            log_std = self.tf_log_std.numpy()
            self.log_std.set_value(log_std)
        else:
            # print("load complete")
            log_std = self.log_std.data
            self.tf_log_std = tf.Variable(log_std)

    @tf.function
    def noise_and_prob(self):
        noise = self.distribution.sample()
        prob = self.distribution.prob(noise)

        prob = tf.clip_by_value(prob, 1e-6, 1)
        # prob = tf.squeeze(prob)

        return noise, prob

    @tf.function
    def __call__(self, *args):
        """
        If use @tf.funtion should modify this function.

        :param args: ((obs1,obs2,...., training),(action,))
        :return: model's out (mu,prob,other info)
        """
        # log_std = self.tf_log_std
        # out = self.run_model(*args[0])
        # mu = out
        # sigma = tf.exp(self.tf_log_std)
        if self.log_std is not None:
            out = self.model(*args[0])
            if isinstance(out, tuple):
                mu = out[0]
            else:
                mu = out
            sigma = tf.exp(self.tf_log_std)
        else:
            out = self.model(*args[0])
            mu = out[0]
            sigma = tf.exp(out[1])

        # action = args[1]
        pi = tfp.distributions.Normal(mu, sigma)
        prob = pi.prob(*args[1])
        prob = tf.clip_by_value(prob, 1e-6, 1)

        # noise, prob = self.noise_and_prob()
        # action = noise + mu*sigma

        if isinstance(out, tuple):
            out = (mu, prob, *out[2:])
        else:
            out = (mu, prob)

        return out

    @tf.function
    def run_model(self, *args):

        return self.model(*args)

    @property
    def get_variable(self):
        # return self.model.trainable_variables + [self.tf_log_std]
        if self.tf_log_std is not None:
            return self.model.trainable_variables + [self.tf_log_std]
        else:
            return self.model.trainable_variables

    def close(self):
        if self.log_std is not None:
            self.log_std.close()
