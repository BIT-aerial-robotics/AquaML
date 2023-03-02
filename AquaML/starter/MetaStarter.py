from mpi4py import MPI
import time
import atexit
import os


def mkdir(path: str):
    """
    create a directory in current path.

    Args:
        path (_type_:str): name of directory.

    Returns:
        _type_: str or None: path of directory.
    """
    current_path = os.getcwd()
    # print(current_path)
    path = os.path.join(current_path, path)
    if not os.path.exists(path):
        os.makedirs(path)
        return path
    else:
        None


class MetaStarter:
    def __init__(self,
                 meta_algo,
                 meta_parameter,
                 inner_algo_starter,
                 inner_parameter,
                 # num_inner_algo,
                 name: str = None,
                 mpi_comm=None,
                 computer_type: str = 'PC',
                 ):

        """
        In meta learning, the name of inner loop will be allocated by the meta algorithm.

        In our meta learning, meta algorithm can directly visit the data
        pool and parameters of inner algorithm.

        As for the neural network, the meta algorithm get the weights via load_weights().

        """
        mkdir(name)
        self.name = name
        self.num_inner_algo = meta_algo['num_inner_algo']
        prefix_inner_algo_name = 'inner'
        self.mpi_comm = mpi_comm
        inner_parameter['meta_flag'] = True
        inner_parameter['computer_type'] = computer_type
        inner_parameter['prefix_name'] = name

        # It should create inner algorithm first, then create meta algorithm.

        if mpi_comm is not None:
            self.total_threads = MPI.COMM_WORLD.Get_size()
            self.thread_id = MPI.COMM_WORLD.Get_rank()
            if self.thread_id == 0:
                pass
            else:
                inner_algo_name = prefix_inner_algo_name + str(self.thread_id - 1)
                inner_parameter['name'] = inner_algo_name

                self.inner_algos = inner_algo_starter(**inner_parameter)
            self.inner_algos = None
        else:
            self.inner_algos = {}
            # allocate name and parameters for inner algorithm
            for i in range(self.num_inner_algo):
                inner_algo_name = prefix_inner_algo_name + str(i)

                inner_parameter['name'] = inner_algo_name

                algo = inner_algo_starter(**inner_parameter)

                # after this, the inner algorithm will be created.
                self.inner_algos[inner_algo_name] = algo

        # Then create meta algorithm
        if mpi_comm is not None:
            if self.thread_id == 0:
                meta_parameter['name'] = name
                meta_parameter['computer_type'] = computer_type

                # instantiate meta algorithm
                self.meta_algo = meta_algo['meta_algo'](**meta_parameter)
        else:
            meta_parameter['name'] = name
            meta_parameter['computer_type'] = computer_type
            self.meta_algo = meta_algo['meta_algo'](**meta_parameter)
