class AquaMLCoordinator:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._init_core(*args, **kwargs)
        return cls._instance

    def _init_core(self, *args, **kwargs):
        self.data_unit_instances = {}  # 用于存储数据单元实例及其信息

    def register_data_unit(self, data_unit_cls):
        """
        装饰器方法，用于拦截数据单元类的实例化过程并记录信息
        :param data_unit_cls: 要注册的数据单元类
        :return: 包装后的类
        """
        def wrapper(*args, **kwargs):
            instance = data_unit_cls(*args, **kwargs)
            instance_info = {
                'class_name': data_unit_cls.__name__,
                'args': args,
                'kwargs': kwargs
            }
            self.data_unit_instances[id(instance)] = instance_info
            print(
                f"Data unit instance of {data_unit_cls.__name__} registered with info: {instance_info}")
            return instance
        return wrapper

    def get_data_unit_instance_info(self, instance):
        """
        根据实例获取其注册信息
        :param instance: 数据单元实例
        :return: 实例的注册信息，如果未找到则返回 None
        """
        return self.data_unit_instances.get(id(instance))


# 使用示例
coordinator = AquaMLCoordinator()


@coordinator.register_data_unit
class DataUnit:
    def __init__(self, data_unit_info):
        self.data_unit_info = data_unit_info


# 实例化 DataUnit 类
data_unit_instance1 = DataUnit("info1")
data_unit_instance2 = DataUnit("info2")

# 获取实例的注册信息
info1 = coordinator.get_data_unit_instance_info(data_unit_instance1)
info2 = coordinator.get_data_unit_instance_info(data_unit_instance2)

print(f"Info of data_unit_instance1: {info1}")
print(f"Info of data_unit_instance2: {info2}")
