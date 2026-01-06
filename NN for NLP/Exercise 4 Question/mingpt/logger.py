from collections import defaultdict
from typing import Dict, List




class Logger:
    def __init__(self):
        self.logs = defaultdict(lambda: {"idx": [], "values": []})

    def log(self, key, iter_num, value):
        self.logs[key]["idx"].append(iter_num)
        self.logs[key]["values"].append(value)
