from mpi4py import MPI
import os
from AquaML.node.Node import Node


def mkdir(path):
    current = os.getcwd()
    path = current + '/' + path
    flag = os.path.exists(path)
    if flag is False:
        os.mkdir(path)


class MPIStarter:
    def __init__(self, task_list: list, ty: str = 'rl'):
        """
        The tool can help you create MPI node.
        :param task_list: task_list describe the relationship between each node. {'level1':task_config,'level2':task_config,'data':}
        notice, data is a list that describes data collector. How to define it you can see AquaML.data.DataCollector. level1 task usually is
        set max_processes to tell starter start how many master node, while level 2 doesn't need
        :param ty: type of your deep learning. such as 'rl'. For different learning method it will have different distributed method.
        """
        global_comm = MPI.COMM_WORLD

        global_size = global_comm.Get_size()

        global_rank = global_comm.Get_rank()

        group = global_comm.Get_group()

        level1_size = task_list['level1'].max_processes  # master size

        self.epoch = task_list['epoch']  # every node has epoch

        # Get every node's data size
        if ty == 'rl':
            # this type need previously tell starter how many steps sample at once.
            single_env_args = task_list['env_args']  # how many

            # compute a master have how many sampling nodes
            # global node + master node
            ave_nodes = int(
                (global_size - (1 + task_list['level1'].max_processes)) / task_list['level1'].max_processes)
            data_dic = task_list['data_dic']
            master_total_length = ave_nodes * single_env_args.total_steps
            sampling_total_length = single_env_args.total_steps

            master_data_args = {'data_dic': data_dic, 'total_length': master_total_length}
            sampling_data_args = {'data_dic': data_dic, 'total_length': sampling_total_length}

        # allocate level 1 node
        # the numbers of processes of level 1 is limited by max_processes.
        # TODO: define communicator.
        if level1_size >= global_rank > 0:
            # in every running, level 1 task is always needed.
            task, args, name = task_list['level1'].get_task(global_rank - 1)  # each level sets starts from 0.

            self.node = Node(node_task=task, data_args=master_data_args, epochs=self.epoch, name=name, task_args=args)

        if 'level 2' in task_list and ty == 'rl':
            # get master name
            sub_rank = global_rank - level1_size - 1  # keep all level id starts from 0
            master_id = int(sub_rank / ave_nodes)
            master_name = task_list['level1'].get_name(master_id)
            task, args, name = task_list['level2'].get_task(sub_rank)

            self.node = Node(node_task=task, data_args=sampling_data_args, epochs=self.epoch, name=name, task_args=args,
                             master_name=master_name)

        # create communicator
        # ranks = [i for i in range(global_size)]  # generate index

        # global_master_ranks = [i for i in range(level1_size + 1)]

        # create communicator between lever1 and level2
        if 'level 2' in task_list:
            # TODO: need to specify root node.
            base_sm_ranks = [i for i in range(ave_nodes)]

            # for master node
            if level1_size >= global_rank > 0:
                sm_ranks = (base_sm_ranks + level1_size) * global_rank
                sm_ranks.insert(0, global_rank)
                master_id = global_rank
            elif global_rank > level1_size:
                master_id = int((global_rank - level1_size - 1) / ave_nodes)

                sm_ranks = (base_sm_ranks + level1_size) * master_id
                sm_ranks.insert(0, master_id)
            else:
                master_id = global_rank  # id = 0

            level2_group = group.Incl(sm_ranks)
            level2_comm = global_comm.Create(level2_group)

        # communicator between master and global_node
        master_ranks = [i + 1 for i in range(level1_size)]
        master_ranks.insert(0, 0)
        master_group = group.Incl(master_ranks)
        master_comm = global_comm.Create(master_ranks)
