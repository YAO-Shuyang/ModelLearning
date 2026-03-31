import numpy as np
from typing import Optional

# CANN PI-Driven
class CANN_PI:
    def __init__(
        self, nn: int = 1000, # number of neurons in the network
        n_hid: int = 111, # number of hidden states
        sigma: float = 3.0, # width of the Gaussian function for recurrent weights,
        tau: float = 10, # time constant for the network dynamics,
        retriv_thre: float = 0.2, 
        epsilon: float = 0.1 # noise level for initial state
    ):
        self.nn = nn # number of neurons
        self.n_hid = n_hid # number of hidden states
        self.sigma = sigma
        self.tau = tau
        self.epsilon = epsilon
        self.W0 = self._create_weight_matrix()
        self.WI = np.eye(self.nn) # identity matrix for sensory input weights
        # shuffle the rows to create a new mapping
        self.WS = np.random.permutation(np.eye(self.nn), axis=0) 
        self.r = self._generate_tuning_curve()
        self.rtv_thre = retriv_thre
        
        # initial state of the network based on the tuning curve and some noise
        self.u0 = self.r[:, 0] + np.random.normal(0, self.epsilon, self.nn) 
        self.b = 0 # initial belief state (can be updated based on the network activity)
        self.p = 0 # initial retrieval probability (can be updated based on the belief state)
        self.u = self.u0.copy()
        self.u_rec = None # to store the recurrent states over time for analysis
        self.corr_rec = None # to store correlation values over time for analysis
        self.belief_rec = None # to store belief states over time for analysis
    
    def update_w0(self):
        self.W0 += self.WS @ self.r # update W0 based on the shuffled tuning curves
        self.W0 = self.W0 / np.max(self.W0) # normalize weights to
    
    def _create_weight_matrix(self) -> np.ndarray:
        """Constructs recurrent weigths depending on sensory sequences.

        Returns
        -------
        W : np.ndarray
            Recurrent weight matrix of shape (nn, nn).
        """
        W = np.zeros((self.nn, self.nn))
        for i in range(self.nn):
            for j in range(self.nn):
                distance = j-i
                W[i, j] = np.exp(-distance**2 / (2 * self.sigma**2))
        W = W / np.max(W) # normalize weights to [0, 1]
        return W
    
    def _generate_tuning_curve(self) -> np.ndarray:
        """Generates tuning curves for each neuron based on the weight matrix.

        Returns
        -------
        tuning_curves : np.ndarray
            Tuning curves of shape (nn, nn).
        """
        tuning_curves = np.zeros((self.nn, self.nn))
        for i in range(self.nn):
            tuning_curves[i] = self.W[i] / np.sum(self.W[i])
        return tuning_curves
    
    def fit(self, s: Optional[np.ndarray] = None, a: Optional[np.ndarray] = None):
        """
        Weights of CANN are preconfigured and no fitting required. 
        This method is a placeholder to maintain consistency with other models.
        """
        self._create_weight_matrix()
    
    def predict(self, s: np.ndarray[int]) :
        """Predicts the next state of the network given the current sensory input.

        Parameters
        ----------
        s : np.ndarray[int]
            Current sensory state inputs of shape (n_L,), where n_L is the length 
            of the sequence of sensory inputs.

        Returns
        -------
        np.ndarray[int]
            Next state of the network of shape (nn,).
        """
        self.reset()
        # initial belief state is certain about the first sensory input
        u_rec = np.zeros((self.nn, len(s)+1))
        b_rec = np.zeros((self.nn, len(s)+1))
        p_rec = np.zeros((self.nn, len(s)+1))
        u_rec[:, 0] = self.u
        
        for t in range(1, len(s)):
            self.u = 1/self.tau * (-self.u + self.WI @ self.W0 @ self.r[:, s[t]])
            u_rec[:, t+1] = self.u

        self.u_rec = u_rec 
        return self.u_rec
    
    def step(self, s_curr: int, s_next: int):
        """Performs one step of the network dynamics given the current sensory input.

        Parameters
        ----------
        s_curr : int
            Current sensory state input at time t.
        s_next : int
            Next sensory state input at time t+1.

        Returns
        -------
        np.ndarray[int]
            Updated state of the network of shape (nn,).
        """
        self.u = 1/self.tau * (-self.u + self.W @ self.r[:, s_curr] + self.r[:, s_next])
        p_dis = self.calc_softmax_belief(self.u) # update belief state based on the current state of the network
        self.b = np.argmax(p_dis) # update belief state based on the current state of the network
        self.p = np.max(p_dis) # update retrieval probability based on the current state of the network
        return
    
    def calc_softmax_belief(self, u: np.ndarray, beta: float = 10.0) -> np.ndarray:
        """
        Convert current CANN activity into a probability distribution over locations.
        Correlation between current state and each stored attractor pattern
        is computed, and then transformed into a probability distribution using 
        a softmax function.

        Parameters
        ----------
        u : np.ndarray
            Current network state, shape (nn,)
        beta : float
            Inverse temperature for softmax. Larger beta -> sharper belief.

        Returns
        -------
        belief : np.ndarray
            Probability over candidate locations, shape (nn,)
        """
        scores = np.zeros(self.nn)

        templates = np.vstack([self.u, self.r.T]) # shape (nn, nn+1)
        # Row-wise correlation of u with each template (the first row is u, the rest are templates)
        scores = np.corrcoef(templates)[0, 1:] # correlation of u with each template

        # handle NaNs safely
        scores = np.nan_to_num(scores, nan=-1.0)

        # softmax with numerical stabilization
        z = beta * scores
        z = z - np.max(z)
        exp_z = np.exp(z)
        belief = exp_z / np.sum(exp_z)

        self.corr_rec = scores
        self.belief_rec = belief
        return belief
    
    def reset(self):
        self.u0 = self.r[:, 0] + np.random.normal(0, self.epsilon, self.nn)
        self.u = self.u0.copy()
        self.b = 0 # initial belief state (can be updated based on the network activity)
        self.p = 0 # initial retrieval probability (can be updated based on the belief state)
        self.u_rec = None # to store the recurrent states over time for analysis
        self.b_rec = None # to store belief states over time for analysis
        self.p_rec = None # to store retrieval probabilities over time for analysis