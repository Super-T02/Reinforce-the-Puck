import logging
import os

from torch.utils.tensorboard import SummaryWriter

# start via cli: tensorboard --logdir logs/tensorboard


class TensorboardStatistics:
    def __init__(self, tb_log_dir: str):
        self._logger = logging.getLogger(__name__)
        if not os.path.exists(tb_log_dir):
            os.makedirs(tb_log_dir)

        # Clear the directory
        for file in os.listdir(tb_log_dir):
            file_path = os.path.join(tb_log_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                pass

        self.writer = SummaryWriter(log_dir=tb_log_dir)
        self._global_steps = 0

    def write_tensorboard_statistics(self, step: int, statistics: dict):
        """
        Write statistics to TensorBoard logs.

        Args:
            writer (SummaryWriter): TensorBoard writer instance.
            step (int): The global step count.
            statistics (dict): A dictionary containing scalar values to log.
        """

        self.writer.add_scalar("step/episode", step, self._global_steps)

        for key, value in statistics.items():
            if isinstance(value, (int, float)):  # Ensure value is a scalar
                self.writer.add_scalar(key, value, self._global_steps)
            else:
                self._logger.error(f"Invalid value type for key '{key}': {type(value)}")
        self._global_steps += 1  # use own global step counter, because the step parameter is reset for each episode

    def close(self):
        self.writer.close()
