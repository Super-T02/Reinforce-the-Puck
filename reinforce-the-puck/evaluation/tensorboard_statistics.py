from torch.utils.tensorboard import SummaryWriter

# start via cli: tensorboard --logdir logs


class TensorboardStatistics:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)

    def write_tensorboard_statistics(self, step: int, statistics: dict):
        """
        Write statistics to TensorBoard logs.

        Args:
            writer (SummaryWriter): TensorBoard writer instance.
            step (int): The global step count.
            statistics (dict): A dictionary containing scalar values to log.
        """
        for key, value in statistics.items():
            if isinstance(value, (int, float)):  # Ensure value is a scalar
                self.writer.add_scalar(key, value, step)
            else:
                print(f"Skipping '{key}' - value is not a scalar: {value}")

    def close(self):
        self.writer.close()
