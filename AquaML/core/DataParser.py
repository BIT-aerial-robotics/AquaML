
class DataInfo:
    """
    Information of dateset or buffer.
    """

    def __init__(self, names: tuple, shapes: tuple, dtypes, dataset=None):
        """Data info struct.

        Args:
            names (tuple): data names.
            shapes (tuple): shapes 
            dtypes (tuple, optional): dtypes.
        """

        # TODO: 当前buffer size
        self.shape_dict = dict(zip(names, shapes))

        self.names = names

        if isinstance(dtypes, tuple):
            self.type_dict = dict(zip(names, dtypes))
        else:
            self.type_dict = dict()
            for key in names:
                self.type_dict[key] = dtypes
        if dataset is not None:
            self.dataset_dict = dict(zip(names, dataset))
        else:
            self.dataset_dict = None

    def add_info(self, name: str, shape, dtype):
        """add info.

        Args:
            name (str): name.
            shape (tuple): shape.
            dtype (type): dtype.
        """
        self.shape_dict[name] = shape
        self.type_dict[name] = dtype

        # add element to names
        names = list(self.names)
        names.append(name)
        self.names = tuple(names)