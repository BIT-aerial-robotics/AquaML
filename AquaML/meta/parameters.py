from AquaML.BaseClass import BaseParameter


class RollOutParameter(BaseParameter):

    def __init__(self, max_steps, total_steps, update_interval, summary_episodes, multi_thread_flag=False):
        super().__init__()
        self.max_steps = max_steps
        self.update_interval = update_interval
        self.summary_episodes = summary_episodes
        self.multi_thread_flag = multi_thread_flag
        self.total_steps = total_steps

        self.buffer_size = self.total_steps

