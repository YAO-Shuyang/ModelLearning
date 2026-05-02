import numpy as np
from typing import Optional
from tqdm import tqdm

# CANN PI-Driven
class CANN_PI:
    def __init__(
        self, nn: int = 1000, # number of neurons in the network
        n_hid: int = 111, # number of hidden states
        sigma: float = 3.0, # width of the Gaussian function for recurrent weights,
        tau: float = 10, # time constant for the network dynamics,
        retriv_thre: float = 0.2, 
        dt: float = 0.1,
        epsilon: float = 0.1, # noise level for initial state,
        J1: float = 0.05, # strength of the global inhibition,
        eta: float = 0.01 # learning rate for Hebbian plasticity
    ):
        self.nn = nn # number of neurons
        self.n_hid = n_hid # number of hidden states
        self.sigma = sigma
        self.tau = tau
        self.dt = dt
        self.epsilon = epsilon
        self.J1 = J1
        self.fit()
        self.WI = np.eye(self.nn) # input matrix
        self.rtv_thre = retriv_thre
        self.eta = eta

        
        # initial state of the network based on the tuning curve and some noise
        self.reset()
    
    def R(self, x: float) -> np.ndarray:
        y = np.exp((-(self.T[:, 0] - x)**2) / (2 * self.sigma**2)) * self.T[:, 1]
        return y
    
    def fit(self, s: Optional[np.ndarray] = None, a: Optional[np.ndarray] = None):
        """create a stored attractor map
        Weights of CANN are preconfigured and no fitting required. 
        This method is a placeholder to maintain consistency with other models."""
        self._generate_tuning_curve()
        self.W0 = self._create_weight_matrix()
    
    def recruite(self, n: int, x: float):
        assert 0 <= n < self.nn
        assert 0 <= x < self.n_hid
        self.T[n, 0] = x
        self.T[n, 1] = 1
    
    def is_retriev(self, u: np.ndarray, x: float):
        """Determine retrieval of a stored attractor given the current network 
        state u and target attractor x.
        
        Returns
        -------
        pr : float
            Pearson correlation (r) between the current network state u and the 
            stored attractor x.
        b : float, [0, 1]
            Retrieval score normalized to [0, 1] based on the correlation,
            determined by a sigmoid function of the correlation.
        """
        pr = np.corrcoef(u, self.R(x))[0, 1]
        b = 1 / (1 + np.exp(-10 * (pr - self.rtv_thre)))
        return pr, b
    
    def _create_weight_matrix(self) -> np.ndarray:
        """Constructs recurrent weigths depending on sensory sequences.

        Returns
        -------
        W : np.ndarray
            Recurrent weight matrix of shape (nn, nn).
        """
        W = np.zeros((self.nn, self.nn))
        n_recruited = self.n_recruited
        idx = np.ix_(n_recruited, n_recruited)
        dist_mat = np.abs(np.subtract.outer(n_recruited, n_recruited))
        W[idx] = np.exp(-dist_mat**2 / (2 * self.sigma**2)) 
        W = W / np.max(W) * (1+self.J1) # normalize weights to [0, 1]
        W -= self.J1 # apply global inhibition
        return W
    
    def _generate_tuning_curve(self):
        """Generates tuning curves for each neuron based on the weight matrix.
        Field centers + Amplitudes.
        """
        self.T = np.zeros((self.nn, 2))
        n_recruited = np.random.choice(np.arange(0, self.nn), size=self.n_hid, replace=False)
        assert np.unique(n_recruited).shape[0] == self.n_hid
        
        self.T[n_recruited, 0] = np.arange(self.n_hid)
        self.T[n_recruited, 1] = 1
        self.n_recruited = n_recruited
    
    def reset(self):
        # initial state of the network based on the tuning curve and some noise
        self.u0 = self.R(0) + np.random.normal(0, self.epsilon, self.nn) 
        self.WI = np.eye(self.nn)
        self.WP = np.eye(self.nn)[np.random.permutation(self.nn), :]
        self.W = self.WI.copy()
        
    def update_W(self, b: float):
        """Update WI based on the retrieval belief b"""
        assert b >= 0 and b <= 1
        self.W = b * self.WI + (1 - b) * self.WP
        
    def update_W0(self, u_rec: np.ndarray):
        """Update W0 based on the retrieval belief b"""
        max_fire_neuron = np.argmax(u_rec, axis=0)
        max_fire_neuron = np.argmax(u_rec)
        dts = np.ones(u_rec.shape[1])
        cum_dts = np.cumsum(dts)
        if max_fire_neuron is not None:
            self.W0[max_fire_neuron, :] += self.eta * self.R(self.u[max_fire_neuron])

    
    def predict(self, s: np.ndarray[float], is_plastic: bool = False):
        """Predicts the next state of the network given the current sensory input.

        Parameters
        ----------
        s : np.ndarray[float]
            Current sensory state inputs of shape (n,), where n is the length 
            of the sequence of sensory inputs.

        Returns
        -------
        np.ndarray[float]
            Next state of the network of shape (nn,).
        """
        self.reset()
        # initial belief state is certain about the first sensory input
        u_rec = np.zeros((self.nn, len(s)))
        b_rec = np.zeros(len(s))
        pr_rec = np.zeros(len(s))
        u_rec[:, 0] = self.u0
        pr_rec[0], b_rec[0] = self.is_retriev(self.u0, s[0])
        
        for t in range(1, len(s)):
            self.u = self.u + self.dt * 1/self.tau * (-self.u + self.W0 @ self.W @ self.R(s[t]))
            u_rec[:, t] = self.u
            pr, b = self.is_retriev(self.u, s[t])
            pr_rec[t] = pr
            b_rec[t] = b
            self.update_W(b)
            if is_plastic:
                self.update_W0()

        return u_rec, b_rec, pr_rec
        
    def predict_many_trials(
        self,
        s: np.ndarray[float],
        n_trials: int,
        is_plastic: bool = False
    ):
        self.reset()
        u_rec = np.zeros((self.nn, len(s), n_trials))
        b_rec = np.zeros((len(s), n_trials))
        pr_rec = np.zeros((len(s), n_trials))
        u0 = self.u0.copy()
        u_rec[:, 0, :] = u0[:, None]
        pr_rec[0, :] , b_rec[0, :] = self.is_retriev(u0, s[0])

        for n in tqdm(range(n_trials)):
            self.u = u0.copy()
            for t in range(1, len(s)):
                self.u = self.u + self.dt * 1/self.tau * (-self.u + self.W @ self.W0 @ self.R(s[t])) + np.random.normal(0, self.epsilon, self.nn)
                u_rec[:, t, n] = self.u
                pr, b = self.is_retriev(self.u, s[t])
                pr_rec[t, n] = pr
                b_rec[t, n] = b
                self.update_W(b)
                if is_plastic:
                    self.update_W0()
            
        return u_rec, b_rec, pr_rec
    
    def visualize_energy_landscape(
        self
    ):
        pass
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    
    model = CANN_PI(nn=1000, n_hid=111, sigma=3.0, dt=0.2, tau=2, epsilon=0.05, retriv_thre=0.3)
    u_rec, b_rec, pr_rec = model.predict_many_trials(
        s = np.linspace(10, 110, 500),
        n_trials=10,
        is_plastic=False
    )
    plt.figure(figsize=(6, 3))
    color = sns.color_palette("rainbow", n_colors=u_rec.shape[2])
    for n in range(u_rec.shape[2]):
        plt.plot(pr_rec[:, n], color=color[n])
    plt.ylim(-0.1, 1)
    plt.show()
    
    plt.figure(figsize=(4, 3))
    ax = plt.subplot(111, projection='3d')
    pca = PCA(n_components=3)
    u_rec_3d = pca.fit(u_rec.reshape(u_rec.shape[0], -1).T)
    for n in range(u_rec.shape[2]):
        u_rec_trial = u_rec[:, :, n].T
        u_rec_3d = pca.transform(u_rec_trial)
        ax.plot(u_rec_3d[:, 0], u_rec_3d[:, 1], u_rec_3d[:, 2], c=color[n])
    plt.show()

    plt.figure(figsize=(4, 3))
    ax = plt.subplot(111, projection='3d')
    standard = np.vstack([model.R(s) for s in np.linspace(10, 110, 500)]).T
    standard_3d = pca.fit_transform(standard.T)
    ax.plot(standard_3d[:, 0], standard_3d[:, 1], standard_3d[:, 2], c='k')
    for n in range(u_rec.shape[2]):
        u_rec_trial = u_rec[:, :, n].T
        u_rec_3d = pca.transform(u_rec_trial)
        ax.plot(u_rec_3d[:, 0], u_rec_3d[:, 1], u_rec_3d[:, 2], c=color[n])
    plt.show()