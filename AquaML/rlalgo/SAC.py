from AquaML.rlalgo.BaseRLAlgo import BaseRLAlgo
from AquaML.DataType import RLIOInfo
from AquaML.rlalgo.Parameters import SAC_parameter
import tensorflow as tf
# TODO: remove in the future
from AquaML.rlalgo.BaseModel import BaseModel


class SAC(BaseRLAlgo):
    
    def __init__(self, env,
                 rl_io_info:RLIOInfo, 
                 parameters:SAC_parameter,
                 # TODO: remove in the future
                 # policy is class do not instantiate
                 actor:BaseModel, # base class is BasePolicy
                 qf1:BaseModel, 
                 qf2:BaseModel,
                 vf:BaseModel,
                 computer_type:str='PC',
                 name:str='SAC',
                 level:int=0, 
                 thread_ID:int=-1, 
                 total_threads:int=1,):
        
        super().__init__(
        env=env,
        rl_io_info=rl_io_info,
        name=name,
        update_interval=parameters.update_interval,
        computer_type=computer_type,
        level=level,
        thread_ID=thread_ID,
        total_threads=total_threads,
        )
        
        # TODO:: initialize the network in the future
        #Notice: qf just be used  in main thread, actor for all threads
        if self.level == 0:
            # main thread
            self.actor = actor 
            self.qf1 = qf1
            self.qf2 = qf2
            
            # create target network
            self.target_vf = vf
            
            # copy the weights
            self.copy_weights(self.vf, self.target_vf)
        else:
            self.actor = actor
            
            # None
            self.qf1 = None
            self.qf2 = None
            self.vf = None
            self.target_vf = None
        
        # create the optimizer
        if self.level == 0:
            self.create_optimizer(name='actor', optimizer=self.actor.optimizer, lr=self.actor.learning_rate)
            self.create_optimizer(name='qf1', optimizer=self.qf1.optimizer, lr=self.qf1.learning_rate)
            self.create_optimizer(name='qf2', optimizer=self.qf2.optimizer, lr=self.qf2.learning_rate)
            self.create_optimizer(name='vf', optimizer=self.vf.optimizer, lr=self.vf.learning_rate)
        
        
        # create distribution
        self.create_gaussian_exploration_policy()

    @tf.function
    def train_q_fun(self,q_obs:tuple, v_next_obs:tuple, reward:tf.Tensor, 
                    mask:tf.Tensor, gamma:float):
        """train the q function

        Args:
            q_obs (tuple): q function input.
            v_next_obs (tuple): value function input.
            reward (tf.Tensor): reward.
            mask (tf.Tensor): mask.
            gamma (float): hyperparameter.

        Returns:
            _type_:  dict
        """
        # TODO: support rnn
        
        ys = tf.stop_gradient(reward + gamma*mask*self.target_vf(*v_next_obs))
        
        # optimize qf1
        with tf.GradientTape() as tape1:
            q1_loss = tf.reduce_mean(tf.square(ys - self.qf1(*q_obs)))*0.5
        
        q1_grad = tape1.gradient(q1_loss, self.qf1.trainable_variables)
        self.qf1_optimizer.apply_gradients(zip(q1_grad, self.qf1.trainable_variables))
        
        # optimize qf2
        with tf.GradientTape() as tape2:
            q2_loss = tf.reduce_mean(tf.square(ys - self.qf2(*q_obs)))*0.5
            
        q2_grad = tape2.gradient(q2_loss, self.qf2.trainable_variables)
        self.qf2_optimizer.apply_gradients(zip(q2_grad, self.qf2.trainable_variables))
        
        ret_dict = {
            'qf1_loss':q1_loss,
            'qf1_grad':q1_grad,
            'qf2_loss':q2_loss,
            'qf2_grad':q2_grad,
        }
        
        return ret_dict
        
            
    @tf.function
    def train_v_fun(self,act_obs:tuple, v_obs:tuple,act:tuple):
        """train the value function

        Args:
            act_obs (tuple): actor input observation. 
            v_obs (tuple): value function input observation.
            act (tuple): action tuple, (action, prob)

        Returns:
            _type_: return the loss and gradient
        """
        
        # Notice: all the out put from the network is tuple
        q1 = self.qf1(*act_obs)
        q2 = self.qf2(*act_obs)
        
        minimum_q = tf.minimum(q1, q2)
        
        action, prob = act # action, prob
        
        log_prob = tf.math.log(prob)
        
        # TODO:这里有两种写法
        with tf.GradientTape() as tape:
            v = self.vf(*v_obs)
            v_loss = tf.reduce_mean(
                0.5*tf.square(v - tf.stop_gradient(minimum_q - log_prob))
            )
            
        v_grad = tape.gradient(v_loss, self.vf.trainable_variables)
        
        self.vf_optimizer.apply_gradients(zip(v_grad, self.vf.trainable_variables))
        
        ret_dict = {
            'vf_loss':v_loss,
            'vf_grad':v_grad,
        }
        
        return ret_dict
    
    def train_actor(self):
        pass