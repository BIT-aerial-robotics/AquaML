from AquaML.RobotAPI.APIBase import APIBase, DataModule
import rospy
from functools import partial
import numpy as np

class ROSAPI(APIBase):
    
    # TODO：未来逐步提高此类的功能，貌似很好用。
    
    def __init__(self, node_name:str ,data_module:DataModule,param:dict=None):
        """
        用于初始化机器人的API。

        Args:
            name (str): 实例名称，区分多个机器人。
            data_module (DataModule): 数据传输模块，用于和机器人交互。
            param (dict, optional): 机器人的参数。 Defaults to None.
        """
        super().__init__(node_name, data_module, param)
        
        rospy.init_node(node_name)
        
        self._robot_state = None
        self._robot_control = None
        
        rospy.Rate(self._run_frequency)
        
    ############################### publisher and subscriber toolkits #################################
        
    def add_subscriber(self, topic_name:str, mapping_name:str, data_type, mapping_func):
        """
        添加订阅者。
        
        
        映射函数的格式(mapping_func)：
        
        def mapping_func(msg):
            arr = np.array(msg.data)
            return arr
        
        注意：这个函数的输入是ROS的数据类型，输出是numpy array。

        Args:
            topic_name (str): 订阅的topic名称。
            mapping_name (str): 映射的变量名称。该topic获取的数据如何对应到module中的变量。
            data_type (class): 数据类型。这个是ROS的数据类型，是一个类。
            mapping_func (function): 映射函数。这个函数用于将ROS的数据类型转换为module中的数据类型。
        """
        
        callback = partial(self.callback, var_name=topic_name, mapping_func=mapping_func)
        rospy.Subscriber(topic_name, data_type, callback)
        
        self._state_mapping_dict[topic_name] = mapping_name
    
    def add_publisher(self, topic_name:str, mapping_name:str ,data_type, mapping_func ,queue_size:int=10):
        """
        添加发布者。
        
        映射函数的格式(mapping_func)：
        
        def mapping_func(arr):
            msg = arr
            return msg
        
        注意：这个函数的输入是module中的数据类型，输出是ROS的数据类型。

        Args:
            topic_name (str): 发布的topic名称。
            mapping_name (str): 映射的变量名称。该topic获取的数据如何对应到module中的变量。
            data_type (class): 数据类型。
            mapping_func (function): 映射函数。这个函数用于将module中的数据类型转换为ROS的数据类型。
            queue_size (int, optional): 队列大小. Defaults to 10.
        """
        
        self._action_mapping_dict[topic_name] = mapping_name
        setattr(self, topic_name, rospy.Publisher(topic_name, data_type, queue_size=queue_size))
        self._publisher_mapping_func_dict[topic_name] = mapping_func
        
    def callback(self, msg, var_name:str, mapping_func):
        """
        回调函数。
        
        # TODO: 优化自动生成函数的方式。

        Args:
            msg (class): 数据类型。
            var_name (str): 变量名称。
        """
        
        data = mapping_func(msg)
        
        setattr(self, var_name, data)
        
    ############################### 运行接口部分 #################################
    
    def get_state(self):
        """
        Get the state from the robot.
        
        获取机器人的状态后将其写入到data_module中。
        """
        
        # 获取状态写入锁
        val = self._data_module.robot_state_update_flag.get_data()
        lenth = val.shape[0]
        
        flag = False
        
        if np.sum(val) == lenth:
            # 获取状态
            flag = True
            
            # 重置状态写入锁,只有写入端才能重置锁。
            self._data_module.robot_state_update_flag.reset_zero()
            
                   
        for topic_name, mapping_name in self._state_mapping_dict.items():
            data = getattr(self, topic_name)
            
            if flag:
                self._data_module.robot_state_dict[mapping_name].set_data(data)
                
    def control(self):
        """
        Control the robot.
        
        从data_module中获取控制信号，然后将其发送给机器人。
        """
        
        # TODO: 优化锁的方式，目前的锁方案可能会导致部分数据丢失。
        
        # 获取control_update_flag，等于0的时候才能更新控制信号。
        val = self._data_module.robot_control_update_flag.get_data()
        
        if val[0] == 0:
            # 读取控制信号
            
            for topic_name, mapping_name in self._action_mapping_dict.items():
                data = self._data_module.robot_control_dict[mapping_name].get_data()
                msg = self._publisher_mapping_func_dict[topic_name](data)
                getattr(self, topic_name).publish(msg)
                
    def run(self):
        """
        运行机器人API。
        """
        
        while not rospy.is_shutdown():
            self.get_state()
            self.control()
            
            rospy.sleep(self._run_time)
        