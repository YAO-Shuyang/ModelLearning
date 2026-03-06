import numpy as np
from typing import Dict, List

maze1_graph = {1: [2, 13],2: [1],3: [4, 15],4: [3, 5],5: [4, 6],6: [5, 7, 18],7: [6, 8],8: [7],9: [10, 21],10: [9, 11],11: [10, 12],12: [11, 24],13: [1, 14, 25],14: [13, 26],15: [3, 27],16: [17, 28],17: [16, 18, 29],18: [6, 17],19: [20, 31],20: [19, 21],21: [9, 20],22: [23, 34],23: [22, 24],24: [12, 23,36],25: [13],26: [14, 27],27: [26, 15],28: [16],29: [17, 30],30: [29, 31, 42],31: [30, 19],32: [33, 44],33: [32, 34],34: [22, 33],35: [36],36: [35, 24],37: [38, 49],38: [37, 39],39: [40, 38, 51],40: [39],41: [42],42: [41, 30],43: [55],44: [32, 45],45: [44, 46],46: [45, 47],47: [46, 48],48: [47, 60],49: [37, 61],50: [51, 62],51: [39, 50,52],52: [51],53: [54],54: [53, 55,66],55: [43, 54,67],56: [57, 68],57: [56, 58],58: [57, 59],59: [58, 60],60: [48, 59],61: [49, 73],62: [50, 74],63: [64, 75],64: [63, 65],65: [64, 66],66: [54, 65],67: [55, 79],68: [56, 69],69: [68, 70],70: [69, 71],71: [70, 72],72: [71, 84],73: [61, 85],74: [62, 75],75: [74, 63],76: [77, 88],77: [76, 89],78: [79, 90],79: [78, 67],80: [81, 92],81: [80, 82],82: [81, 94],83: [95, 84],84: [72, 83, 96],85: [73, 97],86: [98],87: [88, 99],88: [87, 76],89: [77, 101],90: [78, 91],91: [90, 103],92: [80, 104],93: [105],94: [82, 95,106],95: [83, 94],96: [84],97: [85, 98, 109],98: [97, 86],99: [87, 100],100: [99, 112],101: [89, 102],102: [101, 114],103: [91, 104],104: [103, 92],105: [93, 106],106: [105, 94],107: [108, 119],108: [107, 120],109: [97, 110,121],110: [109, 122],111: [112, 123],112: [100, 111],113: [114, 125],114: [113, 102],115: [116, 127],116: [115, 117],117: [116, 129],118: [119, 130],119: [118, 107],120: [108],121: [109, 133],122: [110, 123],123: [111, 122],124: [125, 136],125: [124, 113],126: [127, 138],127: [115, 126],128: [129, 140],129: [128, 117,141],130: [118, 131,142],131: [130, 132],132: [131, 144],133: [121, 134],134: [133, 135],135: [134],136: [124, 137],137: [136, 138],138: [137, 126],139: [140],140: [139, 128],141: [129, 142],142: [130, 141,143],143: [142],144: [132]}

CP_DSP1 = {
    0: np.array([1,13,14,26,27,15,3,4,5,6,18,17,29,30,31,19,20,21,9,10,11,12,24,23,22,34,33,32,44,45,46,47,48,60,59,58,57,56,68,69,70,71,72,84,83,95,94,82,81,80,92,104,103,91,90,78,79,67,55,54,66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144],dtype = np.int64),
    1: np.array([23,22,34,33,32,44,45,46,47,48,60,59,58,57,56,68,69,70,71,72,84,83,95,94,82,81,80,92,104,103,91,90,78,79,67,55,54,66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], np.int64),
    2: np.array([66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], np.int64),
    3: np.array([99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], np.int64),
    4: np.array([8,7,6,18,17,29,30,31,19,20,21,9,10,11,12,24,23,22,34,33,32,44,45,46,47,48,60,59,58,57,56,68,69,70,71,72,84,83,95,94,82,81,80,92,104,103,91,90,78,79,67,55,54,66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85, 97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], np.int64),
    5: np.array([93, 105, 106, 94,82,81,80,92,104,103,91,90,78,79,67,55,54,66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], np.int64),
    6: np.array([135,134,133,121,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], np.int64)
}

NXBIN = 12
NYBIN = 12
GOAL_REWARD = 1.0
BRANCH_END_PENALTY = -0.05
FAMILIARITY_PENALTY = -0.01

def check_relation(
    node_1: int | np.ndarray,
    node_2: int | np.ndarray
):
    if isinstance(node_1, int):
        node_1 = np.array([node_1])
    if isinstance(node_2, int):
        node_2 = np.array([node_2])
        
    x1, y1 = (node_1 - 1) % NXBIN, (node_1 - 1) // NXBIN
    x2, y2 = (node_2 - 1) % NXBIN, (node_2 - 1) // NXBIN
    
    dx = x2 - x1
    dy = y2 - y1
    
    actions = np.zeros(len(node_1), dtype=np.int64)-1
    
    north_mask = (dx == 0) & (dy == 1)
    east_mask = (dx == 1) & (dy == 0)
    south_mask = (dx == 0) & (dy == -1)
    west_mask = (dx == -1) & (dy == 0)
    actions[north_mask] = 0
    actions[east_mask] = 1
    actions[south_mask] = 2
    actions[west_mask] = 3
    
    if np.where(actions == -1)[0].size > 0:
        raise ValueError(f"Nodes {node_1} and {node_2} are not adjacent.")
    
    return actions
    
def to_action_vec(action: int | np.ndarray):
    if isinstance(action, int):
        action = np.array([action])
        
    action_vec = np.zeros((action.shape[0], 2), np.int64)
    north_mask = action == 0
    east_mask = action == 1
    south_mask = action == 2
    west_mask = action == 3
    
    action_vec[north_mask, :] = [0, 1]
    action_vec[east_mask, :] = [1, 0]
    action_vec[south_mask, :] = [0, -1]
    action_vec[west_mask, :] = [-1, 0]
    
    return action_vec

def check_ego_relation(
    avec_1: int | np.ndarray,
    avec_2: int | np.ndarray
):
    if isinstance(avec_1, int):
        avec_1 = np.array([avec_1])
    if isinstance(avec_2, int):
        avec_2 = np.array([avec_2])
        
    ego_actions = np.zeros(len(avec_1), dtype=np.int64)-1
    
    L = np.array([[0, 1], [-1, 0]], np.int64)
    R = np.array([[0, -1], [1, 0]], np.int64)
    F = np.array([[-1, 0], [0, -1]], np.int64)
    B = np.array([[1, 0], [0, 1]], np.int64)
    
    left_mask = np.sum(np.abs(L @ avec_1.T - avec_2.T), axis=0) == 0
    right_mask = np.sum(np.abs(R @ avec_1.T - avec_2.T), axis=0) == 0
    forward_mask = np.sum(np.abs(F @ avec_1.T - avec_2.T), axis=0) == 0
    backward_mask = np.sum(np.abs(B @ avec_1.T - avec_2.T), axis=0) == 0
    
    ego_actions[left_mask] = 0
    ego_actions[forward_mask] = 1
    ego_actions[right_mask] = 2
    ego_actions[backward_mask] = 3
    return np.concatenate(([-1], ego_actions, np.array([-1])))
    
class MazeEnv(object):
    def __init__(self, graph: Dict[int, List[int]], start_node: int, goal_node: int):
        self.graph = graph
        self.n_nodes = len(graph)
        self.legal_actions = np.zeros((self.n_nodes, 4), dtype=bool)
        self.reward_distribution = np.zeros(self.n_nodes)
        self.branch_ends = None
        self.find_branch_ends(start_node=start_node, goal_node=goal_node)
        
        self._init_legal_actions(start_node)
        self._init_reward_distribution(goal_node)
        
    def _init_legal_actions(self, node: int):
        """If there's wall between node and neighbor, then action is not legal."""
        for node, neighbors in self.graph.items():
            for neighbor in neighbors:
                action = check_relation(node, neighbor)
                self.legal_actions[node-1, action] = True
    
    def find_branch_ends(self, start_node: int, goal_node: int):
        """Find the end node given start and goal nodes."""
        branch_end = []
        for n in self.graph.keys():
            if n != start_node and n != goal_node and len(self.graph[n]) == 1:
                branch_end.append(n)
        self.branch_ends = np.array(branch_end)
    
    def _init_reward_distribution(self, goal_node: int):
        self.reward_distribution[goal_node-1] = GOAL_REWARD
        self.reward_distribution[self.branch_ends-1] = BRANCH_END_PENALTY
        
    def step(self, current_node: int, action: int):
        self.legal_actions = np.where(self.legal_actions[current_node-1] == True)[0]
        assert action in self.legal_actions, f"Action {action} is not legal for node {current_node}."
        
        if action == 0:
            dx, dy = 0, 1
        elif action == 1:
            dx, dy = 1, 0
        elif action == 2:
            dx, dy = 0, -1
        elif action == 3:
            dx, dy = -1, 0
        else:
            raise ValueError(f"Invalid action {action}.")
        
        next_node = current_node + dx + dy * NXBIN
        assert next_node >= 1 and next_node <= self.n_nodes, f"Next node {next_node} is out of bounds."
        
        return next_node
    
    def to_ego_actions(self, nodes_sequences: np.ndarray[np.int64]):
        """
        Transform node sequences to ego view sequences.
        
        Ego_Types:
        
        0: Left Turn
        1: Forward
        2: Right Turn
        3: Backward (Turn around)
        
        """
        actions = check_relation(nodes_sequences[:-1], nodes_sequences[1:])
        action_vec = to_action_vec(actions)
        ego_actions = check_ego_relation(action_vec[:-1], action_vec[1:])
        return ego_actions
    
    