import logging
import os
from pathlib import Path
from datetime import datetime

class Logger:

    _instance = None  # Class variable to store the instance

    def __new__(cls):  # Called before the instance is created
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance  # Return the same instance

    def __init__(self):
        print("Initializing Logger...")
        self.logger = None
        self.log_file = None

    def setup_daily_logger(self, console=False):
        if self.log_file is None:
            base_dir = Path(__file__)
            project_dir = base_dir.parent.parent
            logs_dir = os.path.join(project_dir, 'logs')

            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)

            current_time = datetime.now().strftime("%m_%d_%y_%I_%M_%p")
            log_file = os.path.join(logs_dir, f"{current_time}.log")
            self.log_file = log_file

        """ logging.basicConfig(
            filename=log_file,
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', 
            filemode='w',
            encoding='utf-8' 
        ) """

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')  # Added line number
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        self.logger = logger
        return self.logger