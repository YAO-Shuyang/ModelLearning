import numpy as np

class BaseModel:
    """A base class for all models, providing common functionalities and interfaces."""
    
    def fit(self, *args, **kwargs):
        """Fit the model to the data. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")
        
    def fit_by_trial(
        self, 
        trial_beg: np.ndarray, 
        trial_end: np.ndarray, 
        n_iter: int = 100, 
        term_early: bool = False
    ):
        """Fit the model using the EM algorithm, treating each sequence as a separate trial.
        
        Parameters
        ----------
        trial_beg : np.ndarray
            An array of begin indices for each trial in the data.
        trial_end : np.ndarray
            An array of end indices for each trial in the data.
        n_iter : int, optional
            The maximum number of iterations for the EM algorithm (default is 100).
        term_early : bool, optional
            Whether to terminate early if convergence is achieved (default is False).
        
        Raises
        ------
        AssertionError
            If any of the trial begin or end indices are out of range of the observations.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def predict(self, *args, **kwargs):
        """Make predictions using the fitted model. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def predict_prob(
        self, 
        obs_test: np.ndarray[np.int64], 
        act_test: np.ndarray[np.int64]
    ) -> np.ndarray[np.float64]:
        """Predict the state probabilities/likelihoods based on the test observations 
        and actions."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def retrieve(
        self, 
        obs_test: np.ndarray[np.int64], 
        act_test: np.ndarray[np.int64],
        obs_perf: np.ndarray[np.int64],
        act_perf: np.ndarray[np.int64]
    ) -> np.ndarray[np.int64]:
        """Predict the positions based on the test observations and actions.
        
        Parameters
        ----------
        obs_test : np.ndarray[np.int64]
            Array of test observations.
        act_test : np.ndarray[np.int64]
            Array of test actions.
        obs_perf : np.ndarray[np.int64]
            Array of observations for a perfect trial (e.g., the ideal sequence).
        act_perf : np.ndarray[np.int64]
            Array of actions for a perfect trial (e.g., the ideal sequence).
            
        Returns
        -------
        np.ndarray[np.int64]
            Array indicating whether the predicted state matches the state from the 
            perfect trial (1 for match, 0 for mismatch).
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def retrieve_trial_avg(
        self,
        trial_beg: np.ndarray,
        trial_end: np.ndarray,
        obs_test: np.ndarray[np.int64],
        act_test: np.ndarray[np.int64],
        pos_test: np.ndarray[np.int64],
        obs_perf: np.ndarray[np.int64],
        act_perf: np.ndarray[np.int64],
        n_pos_bin: int = 144
    ):
        """
        Notes
        -----
        `pos_test` should be spatial bin ids starting from 1 to `n_pos_bin` (inclusive). 
        The average retrieval is calculated for each spatial bin across all trials, and 
        the final output is an array of average retrieval values for each spatial bin.
        """
        raise NotImplementedError("Subclasses must implement this method.")