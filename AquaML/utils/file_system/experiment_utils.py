
import os
import datetime

def make_exp_dir(base_dir: str = "runs"):
    """
    创建带时间戳的实验目录。

    Args:
        base_dir (str, optional): 实验目录的根路径. Defaults to "runs".

    Returns:
        str: 创建的实验目录的路径.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(base_dir, now)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir
