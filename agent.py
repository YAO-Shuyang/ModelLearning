import numpy as np
from rtrv_models.data.graph import MazeEnv, maze1_graph, maze2_graph
from typing import Optional

class Agent:
    def __init__(self, graph: MazeEnv = MazeEnv(maze1_graph)):
        self.graph = graph
        self.SN = graph.SN
        self.SF = graph.SF
        self._init_S()
        self.b = 0
        self.ep = None # real position
        self.ip = None # inferred position
        self.q = None # action value
        
    def _init_S(self):
        """Initialize the state space of the agent."""
        self.S = self.SN.copy()
        
    def go(self, b: float):
        self.S = b * self.SF + (1-b) * self.SN
        
    
    def reset(self):
        self.b = 0
        self.ep = None
        self._init_S()