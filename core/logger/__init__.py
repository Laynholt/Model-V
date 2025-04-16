import logging
import colorlog

__all__ = ["get_logger"]

def get_logger(name: str = "trainer") -> logging.Logger:
    """
    Creates and configures a logger with colored level names.
    INFO is light blue, DEBUG is green. Message text remains white.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = colorlog.StreamHandler()
        formatter = colorlog.ColoredFormatter(
            fmt="%(log_color)s[%(levelname)s]%(reset)s %(message)s",
            log_colors={
                "DEBUG":    "green",
                "INFO":     "light_blue",
                "WARNING":  "yellow",
                "ERROR":    "red",
                "CRITICAL": "bold_red",
            },
            style="%"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
