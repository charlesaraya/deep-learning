import logging
import os
from pathlib import Path
from datetime import datetime

class Logger:
    """
    Singleton Logger class for managing application-wide logging.

    This class ensures that all modules share the same logger instance.
    It supports logging to a rotating file and optionally to the console.
    """
    _instance = None

    def __new__(cls):  # Called before the instance is created
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance  # Return the same instance

    def __init__(self):
        self.logger = None

    def setup_logger(self, console=False):
        base_dir = Path(__file__)
        project_dir = base_dir.parent.parent
        logs_dir = os.path.join(project_dir, 'logs')

        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        current_time = datetime.now().strftime("%m_%d_%y_%I_%M_%p")
        log_file = os.path.join(logs_dir, f"{current_time}.log")

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        self.logger = logger
        return self.logger