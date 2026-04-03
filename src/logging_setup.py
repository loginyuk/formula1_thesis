import logging

from src.config import ERRORS_LOG


class RaceContextFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.race = 'N/A'

    def filter(self, record):
        record.race = self.race
        return True


def setup_file_logging(log_path=ERRORS_LOG):
    """
    Configures the root logger with a race-context-aware file handler.
    Returns the filter so the caller can update filter.race as processing advances.
    """
    race_filter = RaceContextFilter()

    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s | %(race)s | %(name)s | %(levelname)s | %(message)s')
    )
    file_handler.addFilter(race_filter)

    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(logging.WARNING)

    return race_filter