
from torch.utils.tensorboard import SummaryWriter
import os
from loguru import logger

class TensorboardManager:
    """Manages TensorBoard logging."""
    def __init__(self, log_dir: str):
        """
        Initializes the TensorboardManager.

        Args:
            log_dir (str): The directory where TensorBoard logs will be stored.
        """
        self.writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard writer initialized. Logging to: {log_dir}")

    def write(self, tag: str, value: float, step: int):
        """
        Writes a scalar value to the TensorBoard log.

        Args:
            tag (str): The name of the scalar value.
            value (float): The value to log.
            step (int): The global step value to associate with this data point.
        """
        self.writer.add_scalar(tag, value, step)

    def close(self):
        """Closes the TensorBoard writer."""
        self.writer.close()
        logger.info("TensorBoard writer closed.")
