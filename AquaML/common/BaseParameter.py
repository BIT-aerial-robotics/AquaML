class BaseParameter:
    """
    The base class of parameter.
    """

    def __init__(self):
        self.meta_parameters = {}

    def add_meta_parameters(self, meta_parameters: dict):
        """
        Set the meta parameters.
        """
        for key, value in meta_parameters.items():
            self.meta_parameters[key] = value
            setattr(self, key, value)

    def add_meta_parameter_by_names(self, meta_parameter_names: tuple or list):
        """
        Set the meta parameters by names.
        """
        dic = {}
        for name in meta_parameter_names:
            value = getattr(self, name)
            dic[name] = value

        self.add_meta_parameters(dic)
        self.meta_parameters = dic

    def update_meta_parameters(self, meta_parameters: dict):
        """
        Update the meta parameters.
        """
        for key, value in meta_parameters.items():
            if key in self.meta_parameters.keys():
                self.meta_parameters[key] = value
                setattr(self, key, value)
                
    def update_parameters(self, parameters: dict):
        """
        Update the meta parameters.
        """
        for key, value in parameters.items():
            if key in self.meta_parameters.keys():
                self.meta_parameters[key] = value
                setattr(self, key, value)

    def update_meta_parameter_by_args_pool(self, args_pool):
        """
        Update the meta parameters by args pool.
        """
        for key, value in self.meta_parameters.items():
            value = args_pool.get_param(key)
            setattr(self, key, value)

    @property
    def meta_parameter_names(self):
        return tuple(self.meta_parameters.keys())
    

if __name__ == '__main__':
    parameter = BaseParameter()

    parameter.add_meta_parameters({'a': 1, 'b': 2})

    parameter.add_meta_parameter_by_names(['a', 'b'])

    print(parameter.meta_parameters)