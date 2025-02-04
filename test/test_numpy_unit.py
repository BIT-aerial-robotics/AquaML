from AquaML import coordinator
from AquaML.data import unitCfg, NumpyUnit
import numpy as np



if __name__ == "__main__":
    # 创建一个numpy数据
    unit_cfg = unitCfg(
        name="test_numpy_unit",
        dtype=np.float32,
        single_shape=(1, 2),
        size=2,
        # mode="numpy",
    )

    numpy_unit = NumpyUnit(unit_cfg)

    status_dict = numpy_unit.getUnitStatusDict()

    # print(status_dict)

    # 写入到yaml文件
    # with open("test_numpy_unit.yaml", "w") as f:
    #     yaml.dump(status_dict, f)

    # 遍历unit_cfg的属性并打印
    # for key, value in unit_cfg.__dict__.items():
    #     print(f"{key}: {value}")
