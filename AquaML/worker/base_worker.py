import abc

class BaseWorker(abc.ABC):
    '''
    Base class for all workers.
     
    '''
    def __init__(self):
        
        pass
        
    @abc.abstractmethod
    def run(self):
        '''
        Run the worker.
        '''