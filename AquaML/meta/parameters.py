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


class MetaGradientParameter(RollOutParameter):
    def __init__(self,
                 actor_ratio,
                 critic_ratio,
                 learning_rate,
                 max_epochs,
                 max_steps,
                 total_steps,
                 batch_size,
                 # update_interval,
                 summary_episodes,
                 optimizer='Adam',
                 multi_thread_flag=False):
        super().__init__(
            max_steps=max_steps,
            total_steps=total_steps,
            update_interval=total_steps,
            summary_episodes=summary_episodes,
            multi_thread_flag=multi_thread_flag
        )
        self.max_epochs = max_epochs
        self.actor_ratio = actor_ratio
        self.critic_ratio = critic_ratio
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.batch_size = batch_size


if __name__ == '__main__':
    parameter = MetaGradientParameter(
        max_steps=100,
        total_steps=1000,
        update_interval=10,
        summary_episodes=10,
        multi_thread_flag=True
    )
