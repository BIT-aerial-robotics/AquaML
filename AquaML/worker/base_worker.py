from typing import TYPE_CHECKING

import abc

if TYPE_CHECKING:
    from AquaML.rl_algo import BaseRLAgent



class BaseWorker(abc.ABC):
    '''
    Base class for all workers.

    '''

    def __init__(self, rl_agent: BaseRLAgent,):
        '''
        Initialize the worker.

        :param rl_agent: The RL agent used in the worker.
        :type rl_agent: BaseRLAgent.
        '''
        self.rl_agent = rl_agent

    @abc.abstractmethod
    def run(self):
        '''
        Run the worker.
        '''
