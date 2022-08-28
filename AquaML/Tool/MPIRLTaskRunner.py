from AquaML import DataManager
from AquaML import TaskArgs
from AquaML import RLPolicyManager
from AquaML.policy.GaussianPolicy import GaussianPolicy
from AquaML import RLWorker
import time
import os
import AquaML as A
from mpi4py import MPI
from AquaML.Tool.RLRecoder import Recoder
import numpy as np


def mkdir(path):
    current = os.getcwd()
    path = current + '/' + path
    flag = os.path.exists(path)
    if flag is False:
        os.mkdir(path)


class TaskRunner:
    def __init__(self, task_args: TaskArgs, actor_policy: GaussianPolicy, critic, algo, work_space: str, env,
                 comm: MPI.COMM_WORLD):
        """
        Run your algorithm.

        :param task_args:
        :param work_space: name or work_space of your task.
        :param actor_policy:
        :param critic:
        :param algo: class
        """

        self.data_manager = None
        self.policy_manager = None
        self.task_args = task_args

        # create dir
        self.work_space = work_space
        self.model_path = work_space + '/' + 'models'  # this is also cache path
        self.logs_path = work_space + '/' + 'logs'
        mkdir(self.work_space)
        mkdir(self.model_path)
        mkdir(self.logs_path)

        self.actor_policy = actor_policy
        self.critic = critic

        self.algo = algo
        self.env = env
        self.comm = comm
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()

        # self.actor_policy.model(np.zeros((1,1,2)))

        if self.rank == 0:
            hierarchical_info = {'hierarchical': A.MAIN_THREAD, 'start_pointer': -1, 'end_pointer': -1}
            self.recoder = Recoder(self.work_space)
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
            self.policy_manager = RLPolicyManager(actor_policy=self.actor_policy, critic_model=self.critic,
                                                  action_info=self.task_args.actor_outputs_info,
                                                  actor_input_info=self.task_args.actor_inputs_info,
                                                  work_space=self.work_space,
                                                  hierarchical=hierarchical_info['hierarchical'],
                                                  actor_is_batch_timestep=self.task_args.training_args.actor_is_batch_timesteps
                                                  )
            self.optimizer = self.algo(algo_param=self.task_args.algo_param, train_args=self.task_args.training_args,
                                       data_manager=self.data_manager, policy=self.policy_manager, recoder=self.recoder)
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
            self.policy_manager = RLPolicyManager(actor_policy=self.actor_policy, critic_model=self.critic,
                                                  action_info=self.task_args.actor_outputs_info,
                                                  actor_input_info=self.task_args.actor_inputs_info,
                                                  work_space=self.work_space,
                                                  hierarchical=hierarchical_info['hierarchical'],
                                                  actor_is_batch_timestep=self.task_args.training_args.actor_is_batch_timesteps
                                                  )
            self.worker = RLWorker(
                env_args=self.task_args.env_args,
                policy=self.policy_manager,
                dara_manager=self.data_manager,
                env=self.env
            )

        # self.barrier = mp.Barrier(self.task_args.env_args.worker_num + 1)
        # self.barrier2 = mp.Barrier(self.task_args.env_args.worker_num + 1)
        # if self.task_args.env_args.worker_num > 1:
        #     self.barrier = mp.Barrier(self.task_args.env_args.worker_num + 1)
        #     self.barrier2 = mp.Barrier(self.task_args.env_args.worker_num + 1)
        # else:
        #     self.barrier = None
        #     self.barrier2 = None
        # main thread initial

        # self.data_manager = DataManager(
        #     obs_dic=task_args.obs_info,
        #     action_dic=task_args.actor_outputs_info,
        #     actor_input_info=task_args.actor_inputs_info,
        #     critic_input_info=task_args.critic_inputs_info,
        #     reward_list=task_args.reward_info,
        #     total_length=task_args.env_args.total_steps,
        #     work_space=work_space,
        #     hierarchical_info=main_thread_hierarchical_info
        # )

        # self.policy_manager = RLPolicyManager(actor_policy=actor_policy, critic_model=critic,
        #                                       action_info=task_args.actor_outputs_info,
        #                                       actor_input_info=task_args.actor_inputs_info, work_space=work_space,
        #                                       hierarchical=main_thread_hierarchical_info['hierarchical'])
        #
        # self.algo = algo(algo_param=task_args.algo_param, train_args=task_args.training_args,
        #                  data_manager=self.data_manager, policy=self.policy_manager)

    # debug mode
    # def run(self):
    #     worker = RLWorker(
    #         env_args=self.task_args.env_args,
    #         policy=self.policy_manager,
    #         dara_manager=self.data_manager,
    #         env=self.env
    #     )
    #
    #     for i in range(self.task_args.algo_param.epochs):
    #         worker.roll()
    #         self.algo.optimize()

    def run(self):
        for i in range(self.task_args.algo_param.epochs):
            if self.rank == 0:
                self.policy_manager.sync(self.model_path)
            else:
                pass
            self.comm.Barrier()

            if self.rank > 0:
                self.policy_manager.sync(self.model_path)
                self.worker.roll()

            self.comm.Barrier()

            if self.rank == 0:
                self.optimizer.optimize()
            self.comm.Barrier()

        self.policy_manager.close()
        self.data_manager.close()
        # optimize_thread.join()

    def sample(self, process_id=0):
        """
        sample thread

        :param process_id:
        :return:
        """
        import tensorflow as tf
        # os.environ["CUDA_VISIBLE_DEVICES"] = "2"

        # print("sampling")

        if process_id == 1:
            raise ValueError("Sampling thread can't create share memory block.")

        self.comm.Barrier()

        self.initial(process_id)
        worker = RLWorker(
            env_args=self.task_args.env_args,
            policy=self.policy_manager,
            dara_manager=self.data_manager,
            env=self.env
        )

        for epoch in range(self.task_args.algo_param.epochs):
            self.policy_manager.sync(self.model_path)
            worker.roll()
            self.comm.Barrier()
        self.data_manager.close()
        self.policy_manager.close()

    def initial(self, process_id=0):
        """
        create data pool and RLPolicyManager for each thread.

        :param process_id: (int) if process_ID = 0, task will not use multi thread.
        :return:
        """
        if process_id == 0:
            hierarchical_info = {'hierarchical': 0, 'start_pointer': -1, 'end_pointer': -1}
        else:
            start, end = self.task_args.env_args.sync(process_id - 1)
            if process_id > 1:
                hierarchical = A.SUB_THREAD
            else:
                hierarchical = A.MAIN_THREAD
                start = -1
                end = -1
            hierarchical_info = {'hierarchical': hierarchical, 'start_pointer': start, 'end_pointer': end}

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
        self.policy_manager = RLPolicyManager(actor_policy=self.actor_policy, critic_model=self.critic,
                                              action_info=self.task_args.actor_outputs_info,
                                              actor_input_info=self.task_args.actor_inputs_info,
                                              work_space=self.work_space,
                                              hierarchical=hierarchical_info['hierarchical'])

        # return data_manager, policy_manager

    def create_optimizer(self):
        # import tensorflow as tf

        # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        # physical_devices = tf.config.experimental.list_physical_devices('GPU')
        # if len(physical_devices) > 0:
        #     for k in range(len(physical_devices)):
        #         tf.config.experimental.set_memory_growth(physical_devices[k], True)
        #         print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))

        # if self.task_args.env_args.worker_num > 1:
        process_id = A.MAIN_THREAD

        self.initial(process_id)
        # print(1)

        optimizer = self.algo(algo_param=self.task_args.algo_param, train_args=self.task_args.training_args,
                              data_manager=self.data_manager, policy=self.policy_manager)
        # print(2)
        for epoch in range(self.task_args.algo_param.epochs):
            self.policy_manager.sync(self.model_path)
            if epoch == 0:
                # print('ok')
                self.comm.Barrier()
            self.comm.Barrier()
            optimizer.optimize()
        self.data_manager.close()
        self.policy_manager.close()
