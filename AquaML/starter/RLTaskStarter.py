from AquaML.BaseClass import BaseStarter
from AquaML.DataType import RLIOInfo
from mpi4py import MPI
import time
import atexit


# TODO: 检查任务级别
class RLTaskStarter(BaseStarter):
    def __init__(self, env,
                 model_class_dict: dict,
                 algo,
                 algo_hyperparameter,
                 meta_flag: bool = False,
                 prefix_name: str = None,
                 mpi_comm=None,
                 computer_type: str = 'PC',
                 name=None,
                 ):
        """
        Start a reinforcement learning task.

        Args:
            env : environment. (must be a inherited class of AquaML.BaseClass.RLBaseEnv)
            model_class_dict (dict): model class dict. {'actor':actor_class, 'critic':critic_class}. 
                                    They must be inherited class of AquaML.BaseClass.RLBaseModel.
            algo_hyperparameter : This is a structure. See AquaML.rl.algo.Parameters.
            meta_flag (bool, optional): meta learning flag. Defaults to False.
            mpi_comm (None or MPI.COMM_WORLD): mpi communicator. If True, use mpi to run the task. Default is False.
            computer_type (str): computer type. Default is 'PC'. It decides the way 
                                 to communicate with each thread.
            name (str, optional): name of the task. Defaults to None. When None, use the algorithm name.
        """

        # parallel information
        # if using mpi
        if mpi_comm is not None:
            # get numbers of threads
            total_threads = MPI.COMM_WORLD.Get_size()
            sample_thread = total_threads - 1

            # check  buffer size can be divided by mpi size
            buffer_size = (algo_hyperparameter.buffer_size / sample_thread) * sample_thread
            algo_hyperparameter.buffer_size = int(buffer_size)
            thread_id = MPI.COMM_WORLD.Get_rank()
            # set thread level
            if thread_id == 0:
                level = 0
                env.close()
            else:
                level = 1
        else:
            total_threads = 1
            thread_id = 0
            level = 0

        # TODO: check logics
        # get actor info from model_class_dict
        # just create, do not build
        actor = model_class_dict['actor']()
        actor_out_info = actor.output_info  # dict
        env.set_action_state_info(actor_out_info, actor.input_name)  # set action state info to env
        del actor  # delete actor

        obs_info = env.obs_info

        self.total_threads = total_threads
        self.thread_id = thread_id
        self.level = level
        self.computer_type = computer_type

        # create rl_io_info
        rl_io_info = RLIOInfo(obs_info=obs_info.shape_dict,
                              obs_type_info=obs_info.type_dict,
                              actor_out_info=actor_out_info,
                              reward_info=env.reward_info,
                              buffer_size=algo_hyperparameter.buffer_size,
                              action_space_type=algo_hyperparameter.action_space_type,
                              )

        # create dict for instancing algorithm
        parallel_args = {
            'total_threads': self.total_threads,
            'thread_id': self.thread_id,
            'level': self.level,
            'computer_type': self.computer_type,
        }

        if name is not None:
            parallel_args['name'] = name

        algo_args = {
            'env': env,
            'rl_io_info': rl_io_info,
            'parameters': algo_hyperparameter,
            'prefix_name': prefix_name,
        }

        model_args = model_class_dict

        algo_args = {**algo_args, **model_args, **parallel_args}

        # create algorithm
        self.algo = algo(**algo_args)

        # create folder
        # self.initial_dir(self.algo.name)

        # initial algorithm
        if meta_flag:
            self.algo.meta_init()
        else:
            self.algo.init()

        self.mini_buffer_size = self.algo.each_thread_mini_buffer_size
        self.update_interval = self.algo.each_thread_update_interval

        # roll out length
        if self.mini_buffer_size == 0:
            self.roll_out_length = self.update_interval
        else:
            self.roll_out_length = self.mini_buffer_size

        self.eval_steps = algo_hyperparameter.eval_episodes * algo_hyperparameter.epoch_length
        self.eval_interval = algo_hyperparameter.eval_interval

        # store key objects
        self.max_epochs = algo_hyperparameter.n_epochs * algo_hyperparameter.display_interval
        self.mpi_comm = mpi_comm

        # config run function

        if mpi_comm is None:
            self.run = self._run_
        else:
            self.run = self._run_mpi_

        atexit.register(self.__del__)

    # single thread
    def _run_(self):
        for i in range(self.max_epochs):
            # self.algo.sync()
            self.algo.worker.roll(self.roll_out_length, test_flag=False)
            self.roll_out_length = self.update_interval
            self.algo.optimize()
            if (i + 1) % self.eval_interval == 0:
                self.algo.worker.roll(self.eval_steps, test_flag=True)
            # self.algo.worker.roll(self.eval_steps, test_flag=True)
            # self.algo.sync()

        self.algo.close()

    def meta_run(self):
        # run one epoch
        self.algo.sync()
        self.algo.worker.roll(self.roll_out_length)
        self.roll_out_length = self.update_interval
        self.algo.optimize()

    def _run_mpi_(self):
        for i in range(self.max_epochs):
            if self.level == 0:
                self.algo.sync()
            else:
                pass
            self.mpi_comm.Barrier()

            if self.level == 1:
                self.algo.sync()
                self.algo.worker.roll(self.roll_out_length)
                self.roll_out_length = self.update_interval

            self.mpi_comm.Barrier()

            if self.thread_id == 0:
                self.algo.optimize()
            self.mpi_comm.Barrier()

        self.algo.close()

    def recreate_data_pool(self):
        if self.mpi_comm is not None:
            if self.thread_id == 0:
                self.algo.clear_data_pool()
            self.mpi_comm.Barrier()
            if self.thread_id > 0:
                self.algo.clear_data_pool()
            self.mpi_comm.Barrier()

            if self.thread_id == 0:
                self.algo.recreate_data_pool()
            self.mpi_comm.Barrier()
            if self.thread_id > 0:
                self.algo.reread_data_pool()
        else:
            self.algo.clear_data_pool()
            self.algo.recreate_data_pool()

    # @atexit.register
    def __del__(self):
        print('delete RLTaskStarter')
        print(self.algo.rl_io_info.data_info.shape_dict)
        self.algo.close()
