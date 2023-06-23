import tensorflow as tf
import datetime
import os


def mkdir(path: str):
    """
    create a directory in current path.

    Args:
        path (_type_:str): name of directory.

    Returns:
        _type_: str or None: path of directory.
    """
    current_path = os.getcwd()
    # print(current_path)
    path = os.path.join(current_path, path)
    if not os.path.exists(path):
        os.makedirs(path)
        return path
    else:
        None


# class NewRecoder:
#     def __init__(self, pointed_name=None):
#         self.log_folder = 'logs'
#         mkdir(self.log_folder)
#         if pointed_name is None:
#             self.logs_folder_1 = self.log_folder + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#         else:
#             self.logs_folder_1 = self.log_folder + '/' + pointed_name
#         mkdir(self.logs_folder_1)
#         self.main_writer = tf.summary.create_file_writer(self.logs_folder_1)

class Recoder:
    def __init__(self, pointed_name):

        # folder for storing the log
        # self.log_folder = log_folder

        # file system like:
        # log_folder\
        #  20210101-000000\
        #    main

        self.folder = pointed_name

        # file name
        # if pointed_name is None:
        #     self.logs_folder_1 = self.log_folder + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # else:
        #     self.logs_folder_1 = self.log_folder + '/' + pointed_name

        self.main_writer = tf.summary.create_file_writer(os.path.join(self.folder, 'main'))
        self.max_writer = tf.summary.create_file_writer(os.path.join(self.folder, 'max'))
        self.min_writer = tf.summary.create_file_writer(os.path.join(self.folder, 'min'))

    def record_scalar(self, scalar_dict, step):

        for key, value in scalar_dict.items():
            if 'max' in key:
                with self.max_writer.as_default():
                    tf.summary.scalar(key[:-4], value, step=step)
            elif 'min' in key:
                with self.min_writer.as_default():
                    tf.summary.scalar(key[:-4], value, step=step)
            else:
                with self.main_writer.as_default():
                    tf.summary.scalar(key, value, step=step)

    def record_model_weight(self, model_dict, step):
        for key, value in model_dict.items():
            for weight in value.weights:
                with self.main_writer.as_default():
                    tf.summary.histogram(key + '/' + weight.name, weight, step=step)

    def save_checkpoint(self, model_dict: dict, epoch, checkpoint_dir):

        dir_path = os.path.join(checkpoint_dir, str(epoch))

        mkdir(dir_path)

        for key, value in model_dict.items():
            name = key + '.h5'
            file_path = os.path.join(dir_path, name)
            value.save_weights(file_path, overwrite=True)

    # record histogram
    # def record_histogram(self, name: str, data: tf.Tensor, step: int, prefix: str = ''):
    #     """
    #
    #     Record histogram to tensorboard.
    #     Always used to show grad and weights.
    #
    #     Example:
    #     recoder.record_histogram('grad', grad, step, prefix='critic')
    #     In the tensorboard, the name of the histogram will be 'critic/grad'
    #
    #     Args:
    #         name (str): name of the histogram
    #         data (tf.Tensor): recorded data
    #         step (int): operation step
    #         prefix (str, optional): prefix of the name. Defaults to ''.
    #                                 Discriminate multiple model. If models have critic, prefix should be 'critic/'.
    #     """
    #     name = prefix + '/' + name
    #     with self.main_writer.as_default():
    #         tf.summary.histogram(name, data, step=step)
    #
    # def record_scalar(self, name: str, data: tf.Tensor, step: int, prefix: str = ''):
    #     """
    #
    #     Record scalar to tensorboard.
    #
    #     Example:
    #     recoder.record_scalar('loss', loss, step, prefix='critic')
    #     In the tensorboard, the name of the scalar will be 'critic/loss'
    #
    #     Args:
    #         name (str): name of the scalar
    #         data (tf.Tensor): data to be recorded
    #         step (int): operation step
    #         prefix (str, optional): prefix of the name. Defaults to ''.
    #     """
    #
    #     name = prefix + name
    #
    #     with self.main_writer.as_default():
    #         tf.summary.scalar(name, data, step=step)
    #
    # def record(self, record_dict: dict, step: int, prefix: str = ''):
    #
    #     '''
    #     Record the data to tensorboard  from a dict.
    #
    #     Example:
    #
    #     record_dict = {'critic_loss': critic_loss,'critic_grad': critic_grad,
    #     'critic_var': critic_var, 'actor_loss': actor_loss,
    #     'actor_grad': actor_grad, 'actor_var': actor_var}
    #
    #     recoder.record(record_dict, step, prefix='PPO')
    #
    #     The content of the tensorboard will be:
    #     scalar: PPO/critic_loss, PPO/actor_loss
    #     histogram: PPO/critic_grad/layer_name, PPO/critic_weight/layer_name,
    #                PPO/actor_grad/layer_name, PPO/actor_weight/layer_name
    #
    #     Args:
    #     record_dict (dict): dict of data to be recorded.
    #     step (int): operation step
    #     prefix (str, optional): prefix of the name. Defaults to ''.
    #
    #     '''
    #
    #     for name, key in record_dict.items():
    #         if 'grad' in name:
    #             var_s = record_dict[name[:-4] + 'var']
    #             grad_s = key
    #
    #             for grad, var in zip(grad_s, var_s):
    #                 # record grad
    #                 self.record_histogram(var.name, grad, step, prefix=prefix + '/' + name)
    #
    #                 # record weights
    #                 # self.record_histogram(var.name, var, step, prefix=prefix + '/' + name[:-4] + 'weight')
    #         elif 'var' in name:
    #             var_s = record_dict[name[:-4] + 'var']
    #             grad_s = key
    #             for grad, var in zip(grad_s, var_s):
    #                 # record weights
    #                 self.record_histogram(var.name, var, step, prefix=prefix + '/' + name[:-4] + 'weight')
    #         else:
    #             # record scalar
    #             self.record_scalar(name, key, step, prefix=prefix + '/')
    #
    # def record_weight(self, model, step, prefix=''):
    #     """
    #     Record the weights of the model.
    #     """
    #     for var in model.trainable_variables:
    #         self.record_histogram(var.name, var, step, prefix=prefix)
    #
    # def display_text(self, data_dict: dict):
    #     """
    #     Display text in terminal.
    #
    #     args:
    #         data_dict (dict): dict of data to be displayed.
    #     """
    #
    #     for key, value in data_dict.items():
    #         if 'grad' in key or 'var' in key:
    #             continue
    #         print("{}:{}".format(key, value))
    #
    # def recorde_history_model(self, history_model_log_folder, model_dict: dict, epoch, optimize_info: dict):
    #
    #     # mkdir
    #     dir_path = history_model_log_folder + '/' + str(epoch)
    #     mkdir(dir_path)
    #
    #     # store model
    #     for key, model in model_dict.items():
    #         model.save_weights(dir_path + '/' + key + '.h5', overwrite=True)
    #
    #     # write optimize info
    #     # with open(dir_path + '/' + 'info.txt', 'a') as f:
    #     #     # f.write('{}:{}'.format(key, value))
    #     #     f.write('\r\t')
    #     for key, value in optimize_info.items():
    #         with open(dir_path + '/' + 'info.txt', 'a') as f:
    #             f.write('{}:{}'.format(key, value))
    #             f.write('\n')
