"""Dexterity: base logger class.

Logger mixin class that enables environments to log metrics to tensorboard.
"""

from collections import defaultdict, deque
from typing import Any
import numpy as np
import time
from typing import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import abc
from tabulate import tabulate


class DexterityBaseLogger:
    log_data = {}
    starts = defaultdict(list)
    durations = defaultdict(list)
    plotted_keys = []
    outputs = ['matplotlib']

    @classmethod
    def time(cls, key_or_func: Union[Callable, str]) -> Callable:
        if callable(key_or_func):
            func = key_or_func
            key = func.__name__
        else:
            key = key_or_func

        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                end = time.time()
                cls.starts[key].append(start)
                cls.durations[key].append(end - start)
                
                for output in cls.outputs:
                    getattr(cls, f'to_{output}')(key)
                    plt.pause(0.01)

                return result
            return wrapper
        return decorator
    
    @classmethod
    def to_terminal(cls, key: str, window_size: int = 50) -> None:
        table = []
        for name in cls.durations.keys():
            call_frequency = 1 / np.mean(np.diff(cls.starts[name][-window_size:]))
            duration = np.mean(cls.durations[name][-window_size:])
            fps = 1 / duration
            table.append([name, call_frequency, duration, fps])
        print(tabulate(table, headers=['key', 'call_frequency', 'duration', 'fps'], tablefmt='orgtbl') + '\n')
        
    @classmethod
    def to_matplotlib(cls, key: str) -> None:
        def update_plot(frame: int, window_size: int = 50):
            print("update_plot")
            for i, name in enumerate(cls.durations.keys()):
                cls.axs[i, 0].clear()
                cls.axs[i, 0].set_ylabel(name)
                cls.axs[i, 0].plot(cls.durations[name][-window_size:])
                cls.axs[i, 1].clear()
                cls.axs[i, 1].plot(1 / np.array(cls.durations[name][-window_size:]))
                cls.axs[i, 2].clear()
                cls.axs[i, 2].plot(1 / np.diff(cls.starts[name][-window_size:]))

                if i == 0:
                    cls.axs[i, 0].set_title("Duration [s]")
                    cls.axs[i, 1].set_title("FPS")
                    cls.axs[i, 2].set_title("Call Frequency [Hz]")

            cls.fig.tight_layout()


        print("cls.durations.keys():", cls.durations.keys())
        print("cls.plotted_keys:", cls.plotted_keys)

        # New keys have been added. Create new plot.
        if not set(cls.plotted_keys) == set(cls.durations.keys()):
            plt.close()
            if hasattr(cls, 'fig'):
                del cls.fig
                del cls.axs
                del cls.anim

            cls.fig, cls.axs = plt.subplots(len(cls.durations.keys()), 3, facecolor='#DEDEDE')
            cls.anim = FuncAnimation(cls.fig, update_plot, interval=1000)
            plt.show(block=False)

        if key not in cls.plotted_keys:
            cls.plotted_keys.append(key)

    @staticmethod
    def log(data: Dict[str, Any]) -> None:
        DexterityBaseLogger.log_data = {**DexterityBaseLogger.log_data, **data}

