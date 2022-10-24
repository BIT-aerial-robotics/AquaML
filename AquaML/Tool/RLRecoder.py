import tensorflow as tf
import datetime


# TODO: optimize file system.
class Recoder:
    def __init__(self, workspace: str):
        """
        Recode data. The training information will be stored in workspace/logs

        :param workspace: The name of a task or colony job.
        """

        self.work_space = workspace

        log_dir = self.work_space + "/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M")

        self.main_summary_writer = tf.summary.create_file_writer(log_dir + "/main")

        self.log_dir = log_dir

    def recode_reward(self, reward: dict, epoch: int):
        for key, value in reward.items():
            with self.main_summary_writer.as_default():
                tf.summary.scalar('Traj_info/' + key, value, step=epoch)

    def recode_training_info(self, algo_name: str, training_info: dict, epoch):
        for key, value in training_info.items():
            with self.main_summary_writer.as_default():
                tf.summary.scalar(algo_name + '/' + key, value, step=epoch)

    @staticmethod
    def recode_params(filename, params):
        dic = params.args
        with open(filename, 'w') as f:
            [f.write('{0},{1}\n'.format(key, value)) for key, value in dic.items()]

