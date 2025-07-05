import abc
from AquaML.core.coordinator import coordinator


class BaseRLAgent(abc.ABC):

    def __init__(self):

        pass

    @abc.abstractmethod
    def _getAction(self, state: dict) -> dict:
        '''
        Get the action from the state.

        TODO: split the function in the future.

        :param state: The state needed to get the action.
        :type state: dict{str: tensor or numpy.array} or tensor or numpy.array.

        :return: The action to take.
        :rtype: dict{str: tensor or numpy.array}.
        '''
        raise NotImplementedError

    def getAction(self, state: dict, model_name: str = 'actor') -> dict:
        '''
        This function get corresponding state from coordinator and input it to _getAction function.

        :param state: The state needed to get the action.
        :type state: dict{str: tensor or numpy.array} or tensor or numpy.array.

        :param model_name: The model name.
        :type model_name: str.

        :return: The action to take.
        :rtype: dict{str: tensor or numpy.array}.
        '''

        # get the model from coordinator
        model_dict = coordinator.getModel(model_name)

        # get corresponding state via policy status
        input_names = model_dict['status'].input_names

        input_states = {}

        for input_name in input_names:
            input_states[input_name] = state[input_name]

        return self._getAction(input_states)
