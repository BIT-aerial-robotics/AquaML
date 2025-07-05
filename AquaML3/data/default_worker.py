from .base_worker import BaseWorker
from AquaML.core.coordinator import coordinator


class DefaultWorker(BaseWorker):
    '''
    Default worker for the environment.

    This worker is used to interact with the environment via single process.
    '''

    def __init__(self):
        '''
        Initialize the DefaultWorker.
        '''
        super(DefaultWorker, self).__init__()

        self.reset_flag_ = False

        self.observation_: dict = None

    def run(self, rollout_steps: int):

        if not self.reset_flag_:
            self.reset_flag_ = True
            self.observation_ = self.env_.reset()

        for step in range(rollout_steps):
            actions = self.agent_.getAction(self.observation_)
            next_observation, reward, done, truncated, info = self.env_.step(
                actions)
            self.observation_ = next_observation
        
        
