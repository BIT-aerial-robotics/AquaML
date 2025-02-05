from .base_worker import BaseWorker
from AquaML import coordinator


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

    def run(self, rollout_steps: int):
        pass
