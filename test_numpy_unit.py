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
        mode="numpy",
    )

    numpy_unit = NumpyUnit(unit_cfg)
