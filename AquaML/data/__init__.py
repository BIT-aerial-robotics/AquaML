'''
AquaML所有的数据类都存储在data文件夹下。

这里的数据比较大，一般需要进行传输,或者在分布式里面需要同步的数据。
'''
from .cfg_status import unitCfg
from .numpy_unit import NumpyUnit
from .tensor_unit import TensorUnit
