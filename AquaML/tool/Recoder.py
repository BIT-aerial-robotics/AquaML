import tensorflow as tf
import datetime

class Recoder:
    def __init__(self, log_folder:str,):
        
        # folder for storing the log
        self.log_folder = log_folder
        
        # file system like:
        # log_folder\
        #  20210101-000000\
        #    main
        
        # file name
        self.logs_folder_1 =  self.log_folder + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        self.main_writer = tf.summary.create_file_writer(self.file_name+'/main')
        
    # record histogram
    def record_histogram(self, name:str, data:tf.Tensor, step:int, prefix:str=''):
        """
        
        Record histogram to tensorboard.
        Always used to show grad and weights.
        
        Example:
        recoder.record_histogram('grad', grad, step, prefix='critic')
        In the tensorboard, the name of the histogram will be 'critic/grad'

        Args:
            name (str): name of the histogram
            data (tf.Tensor): recorded data
            step (int): operation step
            prefix (str, optional): prefix of the name. Defaults to ''.
                                    Discriminate multiple model. If models have critic, prefix should be 'critic/'.
        """
        name = prefix + '/' + name
        with self.main_writer.as_default():
            tf.summary.histogram(name, data, step=step) 
    
    def record_scalar(self, name:str, data:tf.Tensor, step:int, prefix:str=''):
        """
        
        Record scalar to tensorboard.
        
        Example:
        recoder.record_scalar('loss', loss, step, prefix='critic')
        In the tensorboard, the name of the scalar will be 'critic/loss'

        Args:
            name (str): name of the scalar
            data (tf.Tensor): data to be recorded
            step (int): operation step
            prefix (str, optional): prefix of the name. Defaults to ''.
        """
        
        name = prefix + name
        
        with self.main_writer.as_default():
            tf.summary.scalar(name, data, step=step) 
    
    def record(self, record_dict:dict, step:int, prefix:str=''):
        
        '''
        Record the data to tensorboard  from a dict.
        
        Example:
        
        record_dict = {'critic_loss': critic_loss,'critic_grad': critic_grad, 
        'critic_var': critic_var, 'actor_loss': actor_loss, 
        'actor_grad': actor_grad, 'actor_var': actor_var}
        
        recoder.record(record_dict, step, prefix='PPO')
        
        The content of the tensorboard will be:
        scalar: PPO/critic_loss, PPO/actor_loss
        histogram: PPO/critic_grad/layer_name, PPO/critic_weight/layer_name, 
                   PPO/actor_grad/layer_name, PPO/actor_weight/layer_name
        
        Args:
        record_dict (dict): dict of data to be recorded.
        step (int): operation step
        prefix (str, optional): prefix of the name. Defaults to ''.
        
        '''
        
        
        
        for name, key in record_dict.items():
            if 'grad' in name:
                var_s = record_dict[name[:-4]+'var']
                grad_s = key
                
                for grad, var in zip(grad_s, var_s):
                    # record grad
                    self.record_histogram(var.name, grad, step, prefix=prefix+'/'+name)
                    
                    # record weights
                    self.record_histogram(var.name, var, step, prefix=prefix+'/'+name[:-4]+'weight')
            else:
                # record scalar
                self.record_scalar(name, key, step, prefix=prefix)
                    
            
 