from AquaML import coordinator
from AquaML.data import unitCfg, NumpyUnit
import numpy as np
from AquaML.file_system import DefaultFileSystem

if __name__ == "__main__":

    # 初始化文件系统
    file_system = DefaultFileSystem(workspace_dir="test_workspace")
    file_system.initFolder()
    coordinator.registerRunner("test_runner")

    unit_cfg = unitCfg(
        name="test_numpy_unit",
        dtype=np.float32,
        single_shape=(1, 2),
        size=2,
        # mode="numpy",
    )

    numpy_unit = NumpyUnit(unit_cfg)

    unit_cfg2 = unitCfg(
        name="test_numpy_unit2",
        dtype=np.float32,
        single_shape=(1, 2),
        size=2,
        # mode="numpy",
    )

    numpy_unit2 = NumpyUnit(unit_cfg2)

    coordinator.saveDataUnit()
