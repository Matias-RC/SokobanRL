import numpy as np
from collections import defaultdict
import random

InitalLibrary = ((-1,0),(1, 0),(0,-1),(0, 1))

class dreamCoder():
    def __init__(self, q, pi, L):
        self.solution_cache = {}
        self.posWalls = None
        self.posGoals = None
        self.q = q
        self.pi = pi
        self.L = L
    
    