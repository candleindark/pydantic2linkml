from enum import auto

from ..tools import StrEnum


class LogLevel(StrEnum):
    @staticmethod
    def _generate_next_value_(name, _start, _count, _last_values):
        return name

    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()
