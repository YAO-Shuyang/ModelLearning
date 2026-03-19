import pickle
import numpy as np
import os

def load_data(mouse: int, parad: str = "DSP") -> dict[str, np.ndarray]:
    """Load data for a given mouse and paradigm.
    
    Parameters
    ----------
    mouse : int
        Mouse ID, only [10212, 10224, 10227, 10232] are valid.
    parad : str, optional
        Paradigm name, either "DSP" or "Pretrain", by default "DSP".
    
    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing the loaded data. Keys include:
        - "MouseID": array of lap numbers for each step point.
        - "Session": array of session numbers for each step point.
        - "Nodes": array of node IDs for each step point.
        - "Speed": array of speeds for each step point.
        - "Route": array of route IDs for each step point.
        - "Lap": array of lap numbers for each step point.
    """
    filename = f"{parad}_{mouse}.pkl"
    with open(os.path.join(os.path.dirname(__file__), filename), "rb") as f:
        data = pickle.load(f)
    return data