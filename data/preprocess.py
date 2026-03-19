import numpy as np
from rtrv_models.data.graph import MazeEnv, maze1_graph, maze2_graph, CP_DSP1
from dataclasses import dataclass
from typing import Optional

@dataclass
class PreprocessedData:
    id: int
    
    lap_train: np.ndarray[np.int64]
    pos_train: np.ndarray[np.int64]
    all_act_train: np.ndarray[np.int64]
    ego_act_train: np.ndarray[np.int64]
    obs_train: np.ndarray[np.int64]
    
    lap_test: np.ndarray[np.int64]
    route_test: np.ndarray[np.int64]
    pos_test: np.ndarray[np.int64]
    all_act_test: np.ndarray[np.int64]
    ego_act_test: np.ndarray[np.int64]
    obs_test: np.ndarray[np.int64]
    env: Optional[MazeEnv] = MazeEnv(maze1_graph)
    
    def __post_init__(self):
        assert (
            len(self.lap_train) == len(self.pos_train) == len(self.all_act_train)
            == len(self.ego_act_train) == len(self.obs_train)
        )
        assert (
            len(self.lap_test) == len(self.route_test) == len(self.pos_test) == len(self.all_act_test)
            == len(self.ego_act_test) == len(self.obs_test)
        )
        
    def lap_train_durs(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the begin and end indexes of each lap in the training data.
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing two arrays:
            - beg: array of begin indices for each lap in the training data.
            - end: array of end indices for each lap in the training data.
        """
        dlap = np.where(np.diff(self.lap_train) != 0)[0]+1
        beg, end = np.concatenate(([0], dlap)), np.concatenate((dlap, [len(self.lap_train)]))
        return beg, end
    
    def lap_test_durs(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the begin and end indexes of each lap in the testing data.
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing two arrays:
            - beg: array of begin indices for each lap in the testing data.
            - end: array of end indices for each lap in the testing data.
        """
        dlap = np.where(np.diff(self.lap_test) != 0)[0]+1
        beg, end = np.concatenate(([0], dlap)), np.concatenate((dlap, [len(self.lap_test)]))
        return beg, end
    
    def get_ideal_seq(self, rt: int, mode: str = 'all') -> tuple[np.ndarray, np.ndarray]:
        """Return ideal action, observation sequences as control.
        Ideal action/obs here means the path connecting the novel starting point and
        the goal without making any detours. This can be viewed as the optimal path 
        animals acquire during training.
        
        Parameters
        ----------
        rt : int
            Route ID. In our task, it ranges over 0 to 6 (included) corresponding to R1 to R7, 
            respectively.
        mode : str, optional
            Action type, either "all" for allocentric actions or 
            "ego" for egocentric actions, by default "all".
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing two arrays:
            - actions: array of action sequences for the specified route in the testing data.
            - observations: array of observation sequences for the specified route in the 
            testing data.
        """
        obs = self.env.obs[CP_DSP1[rt]-1].astype(np.int64)
        if mode == 'all':
            act = self.env.to_all_actions(CP_DSP1[rt])
        elif mode == 'ego':
            act = self.env.to_ego_actions(CP_DSP1[rt])
        else:
            raise ValueError(f"Invalid mode {mode}. Choose 'all' or 'ego'.")
        return act, obs


def preprocess_data(mouse: int) -> PreprocessedData:
    """Preprocess data for a given mouse and paradigm.
    
    Parameters
    ----------
    mouse : int
        Mouse ID, only [10212, 10224, 10227, 10232] are valid.
    Returns
    -------
    PreprocessedData
        An instance of the PreprocessedData class containing the preprocessed data. \\
        `Attributes` include:
        - id: Mouse ID.
        - lap_train: Array of lap numbers for the training data.
        - pos_train: Array of node positions for the training data.
        - all_act_train: Array of allocentric actions for the training data.
        - ego_act_train: Array of egocentric actions for the training data.
        - obs_train: Array of observations for the training data.
        - lap_test: Array of lap numbers for the testing data.
        - route_test: Array of route IDs for the testing data.
        - pos_test: Array of node positions for the testing data.
        - all_act_test: Array of allocentric actions for the testing data.
        - ego_act_test: Array of egocentric actions for the testing data.
        - obs_test: Array of observations for the testing data.
        - env: MazeEnv instance representing the maze environment.
        
        `Functionality` includes:
        - lap_train_durs(): Method to calculate the begin and end indexes of each lap 
        in the training data.
        - lap_test_durs(): Method to calculate the begin and end indexes of each lap 
        in the testing data.
        - get_ideal_seq(rt, mode): Method to return ideal action and observation 
        sequences for a specified route in the testing data, where rt is the route 
        ID and mode specifies the action type (allocentric or egocentric).
    """
    assert mouse in [10212, 10224, 10227, 10232], f"Invalid mouse ID {mouse}. Valid IDs are [10212, 10224, 10227, 10232]."
    from ._io import load_data
    
    data_train = load_data(mouse, "Pretrained")
    data_test = load_data(mouse, "DSP")
    
    MAZE1 = MazeEnv(maze1_graph)
    MAZE2 = MazeEnv(maze2_graph)
    
    # Preprocess training data
    all_act = [] # Allocentric Actions
    # Allocentric actions include:
    # 0: North
    # 1: East
    # 2: South
    # 3: West
    # 4: Goal Reached
    
    ego_act = [] # Egocentric Actions
    # Egocentric actions include:
    # 0: Start Moving
    # 1: Left Turn
    # 2: Forward
    # 3: Right Turn
    # 4: Backward (Turn around)
    # 5: Goal Reached
    
    obs = [] # Observations
    # Observations include:
    # 0: Starting nodes
    # 1: End nodes of branches
    # 2: Straight nodes (nodes with 2 neighbors arranged in a straight line)
    # 3: Turning nodes (nodes with 2 neighbors arranged in a turn)
    # 4: Junction nodes (nodes with 3 or more neighbors)
    # 5: Goal node
    
    dlap = np.where(np.diff(data_train['Lap']) != 0)[0]+1
    beg, end = np.concatenate(([0], dlap)), np.concatenate((dlap, [len(data_train['Lap'])]))
    for b, e in zip(beg, end):
        if data_train['Maze Type'][b] == 1:
            all_act.append(MAZE1.to_all_actions(data_train['Nodes'][b:e]))
            ego_act.append(MAZE1.to_ego_actions(data_train['Nodes'][b:e]))
            obs.append(MAZE1.obs[data_train['Nodes'][b:e]-1].astype(np.int64))
        elif data_train['Maze Type'][b] == 2:
            all_act.append(MAZE2.to_all_actions(data_train['Nodes'][b:e]))
            ego_act.append(MAZE2.to_ego_actions(data_train['Nodes'][b:e]))
            obs.append(MAZE2.obs[data_train['Nodes'][b:e]-1].astype(np.int64))
        else:
            raise ValueError(
                f"Invalid maze type {data_train['Maze Type'][b]} for mouse {mouse} in training data."
            )
    all_act = np.concatenate(all_act, dtype=np.int64)
    ego_act = np.concatenate(ego_act, dtype=np.int64)
    obs = np.concatenate(obs, dtype=np.int64)
    
    # Preprocess testing data
    all_act_test = []
    ego_act_test = []
    obs_test = []
    
    dlap = np.where(np.diff(data_test['Lap']) != 0)[0]+1
    beg, end = np.concatenate(([0], dlap)), np.concatenate((dlap, [len(data_test['Lap'])]))
    for b, e in zip(beg, end):
        all_act_test.append(MAZE1.to_all_actions(data_test['Nodes'][b:e]))
        ego_act_test.append(MAZE1.to_ego_actions(data_test['Nodes'][b:e]))
        obs_test.append(MAZE1.obs[data_test['Nodes'][b:e]-1].astype(np.int64))
        
    all_act_test = np.concatenate(all_act_test, dtype=np.int64)
    ego_act_test = np.concatenate(ego_act_test, dtype=np.int64)
    obs_test = np.concatenate(obs_test, dtype=np.int64)
    
    data = PreprocessedData(
        id=mouse,
        lap_train=data_train['Lap'],
        pos_train=data_train['Nodes'],
        all_act_train=all_act,
        ego_act_train=ego_act,
        obs_train=obs,
        lap_test=data_test['Lap'],
        route_test=data_test['Route'],
        pos_test=data_test['Nodes'],
        all_act_test=all_act_test,
        ego_act_test=ego_act_test,
        obs_test=obs_test,
        env=MAZE1
    )
    return data
