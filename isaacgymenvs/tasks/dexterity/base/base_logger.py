"""Dexterity: base logger class.

Logger mixin class that enables environments to log metrics to tensorboard.
"""

from typing import *


class DexterityBaseLogger:

    def log(self, data: Dict[str, Any]) -> None:
        if not hasattr(self, "log_data"):
            self.log_data = {}
        self.log_data = {**self.log_data, **data}

