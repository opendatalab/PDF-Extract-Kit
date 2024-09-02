import os
import pytz
import logging
import logging.config
from datetime import datetime

# Define the absolute path for the log file
log_directory = os.path.dirname(os.path.abspath(__file__))
# TODO: Define a suitable log_file_path
log_file_path = os.path.join(log_directory, '../app_logs.log')


class CustomFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, tz=None):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.tz = tz

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, self.tz)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            try:
                s = dt.isoformat(timespec='milliseconds')
            except TypeError:
                s = dt.isoformat()
        return s

# TODO: Add to config file
TIMEZONE: str = 'Europe/Madrid'# "Asia/Shanghai"
timezone = pytz.timezone(TIMEZONE)  # Specify your time zone here


# Basic logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            '()': CustomFormatter,
            'format': '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
            'tz': timezone
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
        },
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': log_file_path,
            'formatter': 'standard'
        },
    },
    'loggers': {
        '': {  # Logger root
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': True
        },
        '__name__': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

def setup_logging(name: str = '__main__'):
    """Configures logging for the entire library."""
    logging.config.dictConfig(LOGGING_CONFIG)
    # Get the logger for this specific module
    logger = logging.getLogger(name)
    return logger

# Call configuration immediately
# setup_logging()


