from AquaML.policy.ReplayPolicy import ReplayPolicy
from AquaML.Tool.ReplayWorker import ReplayWorker
import time
import os
import AquaML as A
from mpi4py import MPI
from AquaML import TaskArgs
from AquaML import DataManager


def mkdir(path):
    current = os.getcwd()
    path = current + '/' + path
    flag = os.path.exists(path)
    if flag is False:
        os.mkdir(path)


class ReplayRunner:
    def __init__(self, task_args: TaskArgs, policy, work_space, env, comm):
        # create dir
        self.work_space = work_space
        mkdir(self.work_space)

        self.task_args = task_args

        self.policy = policy

        self.env = env
        self.comm = comm
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()

        if self.rank == 0:
            hierarchical_info = {'hierarchical': A.MAIN_THREAD, 'start_pointer': -1, 'end_pointer': -1}
            self.data_manager = DataManager(
                obs_dic=self.task_args.obs_info,
                action_dic=self.task_args.actor_outputs_info,
                actor_input_info=self.task_args.actor_inputs_info,
                critic_input_info=self.task_args.critic_inputs_info,
                reward_list=self.task_args.reward_info,
                total_length=self.task_args.env_args.total_steps,
                work_space=self.work_space,
                hierarchical_info=hierarchical_info
            )

        else:
            start, end = self.task_args.env_args.sync(self.rank)
            hierarchical_info = {'hierarchical': A.SUB_THREAD, 'start_pointer': start, 'end_pointer': end}
            time.sleep(10)
            self.data_manager = DataManager(
                obs_dic=self.task_args.obs_info,
                action_dic=self.task_args.actor_outputs_info,
                actor_input_info=self.task_args.actor_inputs_info,
                critic_input_info=self.task_args.critic_inputs_info,
                reward_list=self.task_args.reward_info,
                total_length=self.task_args.env_args.total_steps,
                work_space=self.work_space,
                hierarchical_info=hierarchical_info
            )

            self.worker = ReplayWorker(self.policy, self.data_manager, self.env)

    def run(self):
        if self.rank > 0:
            self.worker.roll()

        self.comm.Barrier()

        if self.rank > 0:
            self.data_manager.save_data()

        self.data_manager.close()