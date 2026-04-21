import numpy as np
from scipy.special import digamma, gammaln
from scipy.stats import skellam


class PoissonFactorizationVI:
    """
    Implements Ruiz & Perez-Cruz (2015) coordinate ascent variational inference model for NCAA basketball.
    """

    def __init__(self, T, L, K1=10, K2=10,
                 s_gamma=1.0, r_gamma=1.0,
                 s_alpha=1.0, r_alpha=0.1,
                 s_beta=1.0,  r_beta=0.1,
                 s_eta=1.0,   r_eta=0.1,
                 s_rho=1.0,   r_rho=0.1):
        self.T, self.L = T, L
        self.K1, self.K2 = K1, K2
        self.s_gamma = s_gamma; self.r_gamma = r_gamma
        self.s_alpha = s_alpha; self.r_alpha = r_alpha
        self.s_beta  = s_beta;  self.r_beta  = r_beta
        self.s_eta   = s_eta;   self.r_eta   = r_eta
        self.s_rho   = s_rho;   self.r_rho   = r_rho


    # Helpers
    def _E(self, shp, rte):
        return shp / rte

    def _Elog(self, shp, rte):
        return digamma(shp) - np.log(rte)

    def _init_params(self, rng):
        """
        Initialize near E[alpha]=E[beta]=E[eta]=E[rho]≈1.87 so that the
        prior expected score E[lambda] ≈ (K1+K2) * 1.87^2 ≈ 70, matching
        typical basketball scores
        """
        T, L, K1, K2 = self.T, self.L, self.K1, self.K2
        target = np.sqrt(70.0 / (K1 + K2))   # ≈ 1.87

        def _shp_rte(shape, target_E):
            noise = 1.0 + 0.1 * rng.standard_normal(shape)
            shp = np.abs(noise)                  # ≈ 1 with noise
            rte = shp / (target_E * np.abs(noise))
            return shp, rte

        # gamma initialised at prior mean (E=1)
        self.gamma_shp = float(self.s_gamma)
        self.gamma_rte = float(self.r_gamma)

        self.alpha_shp, self.alpha_rte = _shp_rte((T, K1), target)
        self.beta_shp,  self.beta_rte  = _shp_rte((T, K1), target)
        self.eta_shp,   self.eta_rte   = _shp_rte((L, K2), target)
        self.rho_shp,   self.rho_rte   = _shp_rte((L, K2), target)


    # Phi update (multinomial weights)
    def _update_phi(self, home, away, h_conf, a_conf, neutral):
        K1, K2 = self.K1, self.K2
        Elog_g = self._Elog(self.gamma_shp, self.gamma_rte)
        Elog_a = self._Elog(self.alpha_shp, self.alpha_rte)
        Elog_b = self._Elog(self.beta_shp,  self.beta_rte)
        Elog_e = self._Elog(self.eta_shp,   self.eta_rte)
        Elog_r = self._Elog(self.rho_shp,   self.rho_rte)

        g_log = np.where(neutral, 0.0, Elog_g)[:, None]
        M = len(home)

        log_phi_H = np.empty((M, K1 + K2))
        log_phi_H[:, :K1] = g_log + Elog_a[home] + Elog_b[away]
        log_phi_H[:, K1:] = g_log + Elog_e[h_conf] + Elog_r[a_conf]

        log_phi_A = np.empty((M, K1 + K2))
        log_phi_A[:, :K1] = Elog_a[away] + Elog_b[home]
        log_phi_A[:, K1:] = Elog_e[a_conf] + Elog_r[h_conf]

        log_phi_H -= log_phi_H.max(axis=1, keepdims=True)
        log_phi_A -= log_phi_A.max(axis=1, keepdims=True)
        phi_H = np.exp(log_phi_H); phi_H /= phi_H.sum(axis=1, keepdims=True)
        phi_A = np.exp(log_phi_A); phi_A /= phi_A.sum(axis=1, keepdims=True)
        return phi_H, phi_A

    # Sequential coordinate ascent sweep
    def _update(self, home, away, h_conf, a_conf, ys_H, ys_A, neutral):
        K1, K2 = self.K1, self.K2
        non_neutral = (~neutral).astype(float)

        #   Step 1: phi (given current theta)  
        phi_H, phi_A = self._update_phi(home, away, h_conf, a_conf, neutral)

        #   Step 2: gamma (given current alpha, beta, eta, rho)  
        E_a = self._E(self.alpha_shp, self.alpha_rte)
        E_b = self._E(self.beta_shp,  self.beta_rte)
        E_e = self._E(self.eta_shp,   self.eta_rte)
        E_r = self._E(self.rho_shp,   self.rho_rte)

        self.gamma_shp = self.s_gamma + (non_neutral * ys_H).sum()
        rate_H = ((E_a[home] * E_b[away]).sum(axis=1) +
                  (E_e[h_conf] * E_r[a_conf]).sum(axis=1))
        self.gamma_rte = self.r_gamma + (non_neutral * rate_H).sum()

        #   Recompute E_g with updated gamma  
        E_g  = self._E(self.gamma_shp, self.gamma_rte)
        g_fac = np.where(neutral, 1.0, E_g)[:, None]   # (M, 1)

        #   Step 3: alpha (given NEW gamma, current beta, eta, rho)  
        a_shp = np.full((self.T, K1), self.s_alpha)
        a_rte = np.full((self.T, K1), self.r_alpha)
        np.add.at(a_shp, home, phi_H[:, :K1] * ys_H[:, None])
        np.add.at(a_shp, away, phi_A[:, :K1] * ys_A[:, None])
        np.add.at(a_rte, home, g_fac * E_b[away])
        np.add.at(a_rte, away, E_b[home])
        self.alpha_shp = a_shp
        self.alpha_rte = a_rte

        #   Recompute E_a  
        E_a = self._E(self.alpha_shp, self.alpha_rte)

        #   Step 4: beta (given NEW gamma, NEW alpha, current eta, rho)  
        b_shp = np.full((self.T, K1), self.s_beta)
        b_rte = np.full((self.T, K1), self.r_beta)
        np.add.at(b_shp, away, phi_H[:, :K1] * ys_H[:, None])
        np.add.at(b_shp, home, phi_A[:, :K1] * ys_A[:, None])
        np.add.at(b_rte, away, g_fac * E_a[home])
        np.add.at(b_rte, home, E_a[away])
        self.beta_shp = b_shp
        self.beta_rte = b_rte

        #   Recompute E_b  
        E_b = self._E(self.beta_shp, self.beta_rte)

        #   Step 5: eta (given NEW gamma, NEW alpha, NEW beta, current rho)  
        e_shp = np.full((self.L, K2), self.s_eta)
        e_rte = np.full((self.L, K2), self.r_eta)
        np.add.at(e_shp, h_conf, phi_H[:, K1:] * ys_H[:, None])
        np.add.at(e_shp, a_conf, phi_A[:, K1:] * ys_A[:, None])
        np.add.at(e_rte, h_conf, g_fac * E_r[a_conf])
        np.add.at(e_rte, a_conf, E_r[h_conf])
        self.eta_shp = e_shp
        self.eta_rte = e_rte

        #   Recompute E_e  
        E_e = self._E(self.eta_shp, self.eta_rte)

        #   Step 6: rho (given NEW gamma, NEW alpha, NEW beta, NEW eta)  
        r_shp = np.full((self.L, K2), self.s_rho)
        r_rte = np.full((self.L, K2), self.r_rho)
        np.add.at(r_shp, a_conf, phi_H[:, K1:] * ys_H[:, None])
        np.add.at(r_shp, h_conf, phi_A[:, K1:] * ys_A[:, None])
        np.add.at(r_rte, a_conf, g_fac * E_e[h_conf])
        np.add.at(r_rte, h_conf, E_e[a_conf])
        self.rho_shp = r_shp
        self.rho_rte = r_rte

        #   Recompute phi with final theta for ELBO evaluation  
        phi_H, phi_A = self._update_phi(home, away, h_conf, a_conf, neutral)
        return phi_H, phi_A

    #                       
    # ELBO (augmented-variable form)
    #                       
    def _elbo(self, home, away, h_conf, a_conf, ys_H, ys_A, neutral, phi_H, phi_A):
        K1, K2 = self.K1, self.K2

        Elog_g = self._Elog(self.gamma_shp, self.gamma_rte)
        Elog_a = self._Elog(self.alpha_shp, self.alpha_rte)
        Elog_b = self._Elog(self.beta_shp,  self.beta_rte)
        Elog_e = self._Elog(self.eta_shp,   self.eta_rte)
        Elog_r = self._Elog(self.rho_shp,   self.rho_rte)

        E_g = self._E(self.gamma_shp, self.gamma_rte)
        E_a = self._E(self.alpha_shp, self.alpha_rte)
        E_b = self._E(self.beta_shp,  self.beta_rte)
        E_e = self._E(self.eta_shp,   self.eta_rte)
        E_r = self._E(self.rho_shp,   self.rho_rte)

        g_log = np.where(neutral, 0.0, Elog_g)[:, None]
        g_fac = np.where(neutral, 1.0, E_g)[:, None]

        # E_q[log p(z | theta)] — using E[z_k] = y * phi_k
        ll  = (ys_H[:, None] * phi_H[:, :K1] * (g_log + Elog_a[home] + Elog_b[away])).sum()
        ll += (ys_H[:, None] * phi_H[:, K1:] * (g_log + Elog_e[h_conf] + Elog_r[a_conf])).sum()
        ll += (ys_A[:, None] * phi_A[:, :K1] * (Elog_a[away] + Elog_b[home])).sum()
        ll += (ys_A[:, None] * phi_A[:, K1:] * (Elog_e[a_conf] + Elog_r[h_conf])).sum()
        lam_H = (g_fac * E_a[home] * E_b[away]).sum(axis=1) + \
                (g_fac * E_e[h_conf] * E_r[a_conf]).sum(axis=1)
        lam_A = (E_a[away] * E_b[home]).sum(axis=1) + \
                (E_e[a_conf] * E_r[h_conf]).sum(axis=1)
        ll -= lam_H.sum() + lam_A.sum()

        # E_q[log p(theta | H)]
        def log_prior(shp, rte, s, r):
            return ((s - 1) * (digamma(shp) - np.log(rte))
                    - r * (shp / rte)
                    + s * np.log(r) - gammaln(s)).sum()

        prior = (log_prior(np.array([self.gamma_shp]), np.array([self.gamma_rte]),
                           self.s_gamma, self.r_gamma)
                 + log_prior(self.alpha_shp, self.alpha_rte, self.s_alpha, self.r_alpha)
                 + log_prior(self.beta_shp,  self.beta_rte,  self.s_beta,  self.r_beta)
                 + log_prior(self.eta_shp,   self.eta_rte,   self.s_eta,   self.r_eta)
                 + log_prior(self.rho_shp,   self.rho_rte,   self.s_rho,   self.r_rho))

        # H[q(theta)]
        def gamma_entropy(shp, rte):
            return (shp - np.log(rte) + gammaln(shp) + (1 - shp) * digamma(shp)).sum()

        entropy = (gamma_entropy(np.array([self.gamma_shp]), np.array([self.gamma_rte]))
                   + gamma_entropy(self.alpha_shp, self.alpha_rte)
                   + gamma_entropy(self.beta_shp,  self.beta_rte)
                   + gamma_entropy(self.eta_shp,   self.eta_rte)
                   + gamma_entropy(self.rho_shp,   self.rho_rte))

        # H[q(z)]
        ph = np.clip(phi_H, 1e-300, 1); pa = np.clip(phi_A, 1e-300, 1)
        z_ent = (-ys_H * (ph * np.log(ph)).sum(axis=1)
                 - ys_A * (pa * np.log(pa)).sum(axis=1)).sum()

        return ll + prior + entropy + z_ent

    #                       
    # Fit
    #                       
    def fit(self, home, away, h_conf, a_conf, ys_H, ys_A, neutral,
            max_iter=10_000, tol=1e-8, seed=0, verbose=True):

        rng = np.random.default_rng(seed)
        self._init_params(rng)

        home    = np.asarray(home,    dtype=int)
        away    = np.asarray(away,    dtype=int)
        h_conf  = np.asarray(h_conf,  dtype=int)
        a_conf  = np.asarray(a_conf,  dtype=int)
        ys_H    = np.asarray(ys_H,    dtype=float)
        ys_A    = np.asarray(ys_A,    dtype=float)
        neutral = np.asarray(neutral, dtype=bool)

        prev_elbo = -np.inf

        for it in range(1, max_iter + 1):
            phi_H, phi_A = self._update(home, away, h_conf, a_conf,
                                        ys_H, ys_A, neutral)

            if it % 10 == 0:
                elbo = self._elbo(home, away, h_conf, a_conf,
                                  ys_H, ys_A, neutral, phi_H, phi_A)
                if prev_elbo > -np.inf:
                    delta = abs((elbo - prev_elbo) / (abs(prev_elbo) + 1e-300))
                    if verbose and it % 100 == 0:
                        print(f"  iter {it:6d}  ELBO={elbo:.2f}  delta={delta:.2e}")
                    if delta < tol:
                        if verbose:
                            print(f"  Converged at iter {it}  ELBO={elbo:.2f}")
                        break
                prev_elbo = elbo

        return self

    #                       
    # Predict win probability (all tournament games are neutral court)
    #                       
    def predict_proba(self, home, away, h_conf, a_conf):
        home   = np.asarray(home,   dtype=int)
        away   = np.asarray(away,   dtype=int)
        h_conf = np.asarray(h_conf, dtype=int)
        a_conf = np.asarray(a_conf, dtype=int)

        E_a = self._E(self.alpha_shp, self.alpha_rte)
        E_b = self._E(self.beta_shp,  self.beta_rte)
        E_e = self._E(self.eta_shp,   self.eta_rte)
        E_r = self._E(self.rho_shp,   self.rho_rte)

        lam_H = ((E_a[home] * E_b[away]).sum(axis=1) +
                 (E_e[h_conf] * E_r[a_conf]).sum(axis=1))
        lam_A = ((E_a[away] * E_b[home]).sum(axis=1) +
                 (E_e[a_conf] * E_r[h_conf]).sum(axis=1))

        probs = np.array([
            1 - skellam.cdf(0, mu1, mu2)
            for mu1, mu2 in zip(lam_H, lam_A)
        ])
        return probs, lam_H, lam_A
