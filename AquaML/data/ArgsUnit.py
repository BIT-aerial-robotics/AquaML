class ArgsUnit:
    def __init__(self, name: str, init_value):
        """
        The smallest unit for managing args.

        :param name: The name of args.
        :param init_value: initial value
        """

        self.name = name
        self.value = init_value

    def __call__(self):
        return self.value

    def set_value(self, value):
        self.value = value
