import numpy as np
from dataclasses import dataclass
from typing import Optional

from rtrv_models.naturecomm_cscg.chmm_actions import CHMM
from rtrv_models.base import BaseModel

@dataclass
class CSCG(BaseModel):
    """A class re-encapsulates a CSCG (clone-structured cognitive graph) model 
    for the convenience for standard testings and comparisons across different 
    models.
    
    Attributes
    ----------
    n_clones : np.ndarray[np.int64]
        Number of clones for each observation in the CSCG model.
    act : Optional[np.ndarray[np.int64]]
        Array of actions.
    obs : Optional[np.ndarray[np.int64]]
        Array of observations.
    pos : Optional[np.ndarray[np.int64]]
        Array of positions.
    model : Optional[CHMM]
        The underlying CHMM model.
    """
    n_clones: int
    act: np.ndarray[np.int64]
    obs: np.ndarray[np.int64]
    pos: Optional[np.ndarray[np.int64]] = None
    model: Optional[CHMM] = None
    pseudocount: float = 1e-10
    
    def __post_init__(self):
        self.model = CHMM(
            self.n_clones, 
            x=self.obs, 
            a=self.act, 
            pseudocount=self.pseudocount
        )
        assert self.obs.shape[0] == self.act.shape[0], \
            f"Observation {self.obs.shape[0]} and action {self.act.shape[0]} "\
            f"arrays must have the same length."
    
    def fit(self, n_iter: int = 100, term_early: bool = False) -> list[float]:
        """Fit the CSCG model using the EM algorithm."""
        progression = self.model.learn_em_T(
            x=self.obs, 
            a=self.act, 
            n_iter=n_iter, 
            term_early=term_early
        )
        return progression
        
    def fit_by_trial(
        self, 
        trial_beg: np.ndarray, 
        trial_end: np.ndarray, 
        n_iter: int = 100, 
        term_early: bool = False
    ):
        """Fit the CSCG model using the EM algorithm, treating each sequence 
        as a separate trial."""
        assert np.max(trial_beg) <= self.obs.shape[0], \
            f"Trial begin indices must be within the range of observations."
        assert np.max(trial_end) <= self.obs.shape[0], \
            f"Trial end indices must be within the range of observations."
        
        obs = [self.obs[beg:end] for beg, end in zip(trial_beg, trial_end)]
        act = [self.act[beg:end] for beg, end in zip(trial_beg, trial_end)]
        
        i = 0
        for obs_seq, act_seq in zip(obs, act):
            if i % 20 == 0:
                print(f"Processing trial {i}")

            self.model.learn_em_T(
                x=obs_seq, 
                a=act_seq, 
                n_iter=n_iter, 
                term_early=term_early
            )
            i += 1
            
    def predict(
        self, 
        obs_test: np.ndarray[np.int64], 
        act_test: np.ndarray[np.int64]
    ) -> np.ndarray[np.int64]:
        """Predict the latent state based on the test observations and actions."""
        return self.model.decode(obs_test, act_test)[1].astype(np.int64)
    
    def predict_with_plasticity(
        self,
        obs_test: np.ndarray[np.int64],
        act_test: np.ndarray[np.int64],
    ):
        """Predict the latent state based on the test observations and actions, 
        allowing for plasticity (i.e., updating the model parameters based on 
        the test data)."""
        self.model.learn_em_T(x=obs_test, a=act_test, n_iter=10, term_early=True)
        return self.model.decode(obs_test, act_test)[1].astype(np.int64)
    
    def predict_prob(
        self, 
        obs_test: np.ndarray[np.int64], 
        act_test: np.ndarray[np.int64]
    ) -> np.ndarray[np.float64]:
        """Predict the state probabilities/likelihoods based on the test observations 
        and actions."""
        return self.model.decode(obs_test, act_test)[0].astype(np.float64)
    
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
        state_test = self.predict(obs_perf, act_perf)
        state_pred = self.predict(obs_test, act_test)
        return np.where(state_test - state_pred == 0, 1, 0)
    
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
        the final output is an array of average retrieval values for each spatial bin.\\
            
        Compute data of each route separately.
        """
        assert np.max(trial_beg) < self.obs.shape[0], \
            f"Trial begin indices must be within the range of observations."
        assert np.max(trial_end) < self.obs.shape[0], \
            f"Trial end indices must be within the range of observations."
        
        avg_retrieval = np.zeros(n_pos_bin)
        
        for beg, end in zip(trial_beg, trial_end):
            obs_test_trial = obs_test[beg:end]
            act_test_trial = act_test[beg:end]
            pos_test_trial = pos_test[beg:end]
            
            retrieval_trial = self.retrieve(
                obs_test_trial, 
                act_test_trial, 
                obs_perf, 
                act_perf
            )
            
            for pos in range(1, n_pos_bin+1):
                avg_retrieval[pos-1] += np.mean(retrieval_trial[pos_test_trial == pos])
                
        avg_retrieval /= len(trial_beg)
        return avg_retrieval
