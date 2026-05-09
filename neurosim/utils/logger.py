"""
Logging utilities for the neural simulator.
"""

import logging
import sys
from datetime import datetime

__all__ = ['get_logger', 'SimLogger', 'LOG_LEVELS']

# Custom log level for detailed diagnostic output
DIAGNOSTIC = 5
logging.addLevelName(DIAGNOSTIC, 'DIAGNOSTIC')

LOG_LEVELS = {
    'CRITICAL': logging.CRITICAL,
    'ERROR': logging.ERROR,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'DIAGNOSTIC': DIAGNOSTIC,
}


class SimLogger:
    """
    Logging system for the neural simulator.

    Provides consistent logging across the application with configurable
    levels and formatting.
    """

    _initialized = False
    _loggers = {}
    _console_handler = None
    _log_level = logging.INFO

    @classmethod
    def initialize(cls, level='INFO'):
        """
        Initialize the logging system.

        Parameters
        ----------
        level : str
            The default log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        if cls._initialized:
            return

        cls._log_level = LOG_LEVELS.get(level.upper(), logging.INFO)

        # Create console handler
        cls._console_handler = logging.StreamHandler(sys.stderr)
        cls._console_handler.setLevel(cls._log_level)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        cls._console_handler.setFormatter(formatter)

        cls._initialized = True

    @classmethod
    def set_level(cls, level):
        """Set the global log level."""
        cls._log_level = LOG_LEVELS.get(level.upper(), logging.INFO)
        if cls._console_handler:
            cls._console_handler.setLevel(cls._log_level)
        for logger in cls._loggers.values():
            logger.setLevel(cls._log_level)

    @classmethod
    def get_logger(cls, name):
        """
        Get a logger for the specified module.

        Parameters
        ----------
        name : str
            The module name (typically __name__)

        Returns
        -------
        logger : logging.Logger
            A configured logger instance
        """
        if not cls._initialized:
            cls.initialize()

        if name not in cls._loggers:
            logger = logging.getLogger(name)
            logger.setLevel(cls._log_level)
            logger.addHandler(cls._console_handler)
            logger.propagate = False
            cls._loggers[name] = logger

        return cls._loggers[name]

    @classmethod
    def log(cls, level, message, name='neurosim'):
        """
        Log a message at the specified level.

        Parameters
        ----------
        level : str
            Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        message : str
            The message to log
        name : str
            Logger name
        """
        logger = cls.get_logger(name)
        numeric_level = LOG_LEVELS.get(level.upper(), logging.INFO)
        logger.log(numeric_level, message)

    @classmethod
    def debug(cls, message, name='neurosim'):
        """Log a debug message."""
        cls.log('DEBUG', message, name)

    @classmethod
    def info(cls, message, name='neurosim'):
        """Log an info message."""
        cls.log('INFO', message, name)

    @classmethod
    def warning(cls, message, name='neurosim'):
        """Log a warning message."""
        cls.log('WARNING', message, name)

    @classmethod
    def error(cls, message, name='neurosim'):
        """Log an error message."""
        cls.log('ERROR', message, name)

    @classmethod
    def diagnostic(cls, message, name='neurosim'):
        """Log a diagnostic message (very detailed debug output)."""
        cls.log('DIAGNOSTIC', message, name)


def get_logger(name):
    """
    Get a logger for the specified module.

    Parameters
    ----------
    name : str
        The module name (typically __name__)

    Returns
    -------
    logger : logging.Logger
        A configured logger instance
    """
    return SimLogger.get_logger(name)
