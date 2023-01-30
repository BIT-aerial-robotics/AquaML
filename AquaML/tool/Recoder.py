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

        Args:
            name (str): name of the histogram
            data (tf.Tensor): recorded data
            step (int): operation step
            prefix (str, optional): prefix of the name. Defaults to ''.
                                    Discriminate multiple model. If models have critic, prefix should be 'critic/'.
        """
        name = prefix + name
        with self.main_writer.as_default():
            tf.summary.histogram(name, data, step=step) 
    
    def record_scalar(self, name:str, data:tf.Tensor, step:int, prefix:str=''):
        """
        
        Record scalar to tensorboard.

        Args:
            name (str): name of the scalar
            data (tf.Tensor): data to be recorded
            step (int): operation step
            prefix (str, optional): prefix of the name. Defaults to ''.
        """
        
        name = prefix + name
        
        with self.main_writer.as_default():
            tf.summary.scalar(name, data, step=step) 
            
 