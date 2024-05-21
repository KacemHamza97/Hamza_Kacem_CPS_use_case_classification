import os
import logging
from datetime import datetime


def setup_logger(name):
    """Set up logger with specified name and log to both console and file."""
    # Create a directory for log files
    log_directory = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_directory, exist_ok=True)

    # Define the log file name using the current timestamp
    log_file = datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".log"
    log_file_path = os.path.join(log_directory, log_file)

    # Configure logging
    logging.basicConfig(
        format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
    )

    # Get the logger instance
    logger = logging.getLogger(name)
    return logger
