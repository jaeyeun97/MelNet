import torch
from util import split, interleave

class Solver(object):
    def __init__(self, config):
        self.config = config
