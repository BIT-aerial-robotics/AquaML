from AquaML.data.ArgsUnit import ArgsUnit


class TrainArgs:
    def __init__(self,
                 # lstm训练参数
                 burn_in: int = 0
                 ):
        """
        This args points to each algorithm, it controls pre-process of data.

        :param burn_in: (int) Use some data to initiate networks hidden state.
        """
        self.burn_in = ArgsUnit('burn_in', burn_in)
        self.args = {'burn_in': self.burn_in}  # All the Args class must contain this attribute.
