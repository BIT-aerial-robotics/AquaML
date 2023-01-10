class DataInfo:
    """
    Information of dateset or buffer.
    """
    def __init__(self, names:tuple, shapes:tuple, dtypes, dataset=None):
        """Data info srtuct.

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
        


if __name__ == "__main__":
    a = DataInfo(('obs','critic'),((13,1),(12,1)),float)

    print(a.shape_dict)
