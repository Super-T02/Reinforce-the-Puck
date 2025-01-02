import os
from queue import Queue
from threading import Thread

from torch.utils.tensorboard import SummaryWriter


class TensorboardStatistics:
    def __init__(self, tb_log_dir: str):
        if not os.path.exists(tb_log_dir):
            os.makedirs(tb_log_dir)

        # Clear the directory
        for file in os.listdir(tb_log_dir):
            file_path = os.path.join(tb_log_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception:
                pass

        self.writer = SummaryWriter(log_dir=tb_log_dir)
        self._global_steps = 0

        # Queue and Thread for asynchronous logging
        self._log_queue = Queue()
        self._stop_signal = False
        self._worker_thread = Thread(target=self._log_worker, daemon=True)
        self._worker_thread.start()

    def _log_worker(self):
        """Worker thread for processing log entries."""
        while not self._stop_signal or not self._log_queue.empty():
            try:
                step, statistics = self._log_queue.get(timeout=0.1)
                self._write_to_tensorboard(step, statistics)
                self._log_queue.task_done()
            except Exception:
                pass  # Handle empty queue or other issues gracefully

    def _write_to_tensorboard(self, step: int, statistics: dict):
        """Internal method to write directly to TensorBoard."""

        self.writer.add_scalar("step/episode", step, self._global_steps)

        for key, value in statistics.items():
            if isinstance(value, (int, float)):  # Ensure value is a scalar
                self.writer.add_scalar(key, value, self._global_steps)
            else:
                pass
        self._global_steps += 1  # use own global step counter, because the step parameter is reset for each episode

    def write_tensorboard_statistics_async(self, step: int, statistics: dict):
        """
        Add statistics to the queue for asynchronous logging.
        """
        self._log_queue.put((step, statistics))

    def close(self):
        """
        Close the writer and stop the worker thread.
        """
        self._stop_signal = True
        self._worker_thread.join()
        self.writer.close()
