import time
from AquaML.data.DataPool import DataPool
from AquaML.data.DataCollector import DataCollector
from AquaML.args.ComArgs import NodeInfo
from AquaML.node.MPIComm import MPIComm


class Node:
    def __init__(self, node_task, data_args: dict, epochs, name: str, task_args=None, master_name=None):
        """
        The minimum running task. Initial tensorflow for every thread. In our framework, RLNode is composed by task, datacollector, policy.
        :param node_task: The node will run this task. node_task can be an instance of a class, at this situation, task_args=None. Or node_task is a class.
        :param task_args: see node_task. None or dict.
        """

        self.task_args = task_args

        self.name = name

        if self.task_args is None:
            self.node_task = node_task
        else:
            self.node_task = node_task(**task_args)

        if master_name is not None:
            data_args['name_prefix'] = master_name
        else:
            data_args['name_prefix'] = name

        self.data_collector = DataCollector(**data_args)

        self.node_task.data_collector = self.data_collector

        self.epochs = epochs

    def run(self):
        for i in range(self.epochs):
            # TODO: every task must have sync method
            self.node_task.sync()
            self.node_task.run()
            # TODO: implement data transition method
