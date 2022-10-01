class TaskConfig:
    def __init__(self, task, name: str, task_args=None,initializer=False, master_name=None, max_processes=None):
        """
        Config task about your node.Record the same type task. Give every type a name. Distribute task for
        every node.
        Combination:
                    task  args
                    1       1
                    1       1(vary depend on rank)   current stage you should manually generate
                    1       list
                    list    list
                    1        None (created preview)
        args is a class, when task_args is not list, args must contain initialize(id).

        :param task:
        :param task_args:
        :param name:(str) The same type name. usually choose master
        :param initializer: args have initializer?
        :param master_name: optional, it will help you name process regularly.
        """

        self.name = name

        self.task = task
        self.task_args = task_args

        self.initializer = initializer
        self.master_name = master_name

        self.max_processes = max_processes

    def get_task(self, id):
        """
        id start from 0
        :param id:
        :return: task, args, name
        """
        if isinstance(self.task_args, list):
            task_args = self.task_args[id]
        elif self.task_args is None:
            task_args = None
        else:
            if self.initializer:
                task_args = self.task_args.initializer(id)
            else:
                task_args = self.task_args

        if isinstance(self.task, list):
            task = self.task_args[id]
        else:
            task = self.task

        if self.master_name is None:

            process_name = self.name + '_' + str(id)
        else:
            process_name = self.master_name + '_' + self.name + '_' + str(id)

        return task, task_args, process_name

    def set_master_name(self, master_name):
        # automatically used by MPIStarter

        self.master_name = master_name

    def get_name(self, id):
        if self.master_name is None:
            process_name = self.name + '_' + str(id)
        else:
            process_name = self.master_name + '_' + self.name + '_' + str(id)

        return process_name
