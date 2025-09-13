# Machine Learning as Adaptive Evolution Optimization (AEO)

## Abstract

**Thesis.** Every practical machine-learning procedure is a form of **Adaptive Evolution Optimization**: a closed-loop system where (i) task parameters $\theta$ update to reduce an objective, while (ii) a **genotype** of optimizer coefficients $a$ (learning rate, momentum, regularization, noise, filters, curricula, etc.) **evolves continuously** in response to filtered performance signals. “Evolution” here means **continuous coefficient adaptation**—not reproduction or generations. AEO supplies a compact set of **laws**, a **normal form** for any optimizer, and stability/efficiency guarantees, unifying SGD/Adam, RL, self-supervision, Bayesian/variational learning, diffusion training, and nonstationary settings.

---

## 1) Core Mapping (ML ⇔ AEO)

* **Phenotype (dynamics):** trajectory of $\theta_t$, losses, returns, and stability.
* **Genotype (coefficients):** $a_t\in\mathbb{R}^m$ (e.g., $\alpha,\beta,\lambda,\sigma,$ filter constants, curriculum strengths).
* **Selection (feedback):** filtered observables $s_t$ built from loss, reward, gradient statistics, alignment, uncertainty.
* **Evolution:** closed-loop updates of $a_t$ governed by differentiable laws.

**Normal form of any ML step**

$$
\theta_{t+1}=\theta_t - \alpha_t\,P_t(a_t,s_t)\,g_t \;+\; \xi_t(a_t,s_t),
$$

where $g_t=\nabla_\theta \ell(\theta_t)$ (or a policy/critic gradient, or a subgradient), $P_t$ is a preconditioner (identity, Adam/RMS, natural grad, etc.), and $\xi_t$ is structured noise (SGD minibatch, entropy temperature, parameter noise).

**AEO layer (meta-controller)**

$$
a_{t+1}=\Pi_\Omega\!\big(a_t\odot\exp(\eta_a\,u_t)\big),
\qquad
u_t=\mathcal{U}(s_t,a_t),
$$

with projection/boundary $\Pi_\Omega$, small meta-step $\eta_a$, and smooth drive $u_t$.

---

## 2) Signals (Filtered Observables)

Causal, differentiable filters produce **trend** $T_t$, **variance/stability** $V_t$, **reward-delta** $R_t$, **phase/alignment** $\rho_t=\cos\angle(g_t,g_{t-1})$, and **uncertainty** $S_t$ (e.g., variance of gradients or returns):

$$
s_t=[T_t,V_t,R_t,\rho_t,S_t]=\mathcal{F}_\tau(\ell_t,r_t,g_t).
$$

Filtering time-constants $\tau$ set the **adaptation time scale**.

---

## 3) Laws of AEO (compact)

**AEO-1: Measurement (Filtering).** Coefficients react only to bounded, causal, differentiable signals $s_t$.

**AEO-2: Constitutive Drive.** Combine progress, stability, reward, priors, and constraints:

$$
u_t=W_p\tanh(-T_t/\tau_p)-W_v\,\text{softplus}(V_t/\tau_v)+W_r\tanh(R_t/\tau_r)
+W_c(\rho_t-\rho^\star)-\Gamma(a_t-\bar a)-\nabla_a \psi_\Omega(a_t).
$$

**AEO-3: Evolution (Multiplicative Update).**

$$
a_{t+1}=\Pi_\Omega\!\big(a_t\odot\exp(\eta_a\,u_t)\big),
$$

ensuring positivity/scale-invariance; optional SDE form adds controlled diffusion.

**AEO-4: Exploration Temperature (Noise Law).**

$$
\log\sigma_{t+1}=\log\sigma_t+\eta_\sigma\!\Big(\kappa_s\mathbf{1}\{|T_t|<\varepsilon\}+\kappa_u S_t/S^\star-\kappa_v V_t/V^\star+\kappa_a(A_t-A^\star)\Big).
$$

**AEO-5: Resonance–Stability (Phase Control).**

$$
u_t \leftarrow u_t + W_m(\rho_t-\rho^\star),\quad
\text{if }V_t>V_{\max}\text{ or }|\rho_t|>\rho_{\max}: \; a_{t+1}\!\leftarrow\!\frac{a_{t+1}}{1+\kappa_{\text{shrink}}}.
$$

**AEO-6: Power/Budget (Continuity).** Bound per-step adaptation energy $\|a_{t+1}-a_t\|_M^2\le B$ via a dual variable (smooth trust-region).

**AEO-7: Coupling (Geometry).** Cross-terms couple coefficients (e.g., high momentum ↔ lower $\alpha$):

$$
u_t \leftarrow u_t + C\,\zeta_t,\quad
\zeta_t=[\log\alpha_t,\log\tfrac{1}{1-\beta_t},\log\sigma_t,\log\lambda_t,\dots]^\top.
$$

**AEO-8: Gauge (Effective Step Invariance).** Keep $\gamma_t=\alpha_t\|g_t\|$ near a target:

$$
\log\alpha_{t+1}=\log\alpha_t+\eta_\gamma\Big(\tfrac{\gamma^\star-\gamma_t}{\gamma^\star}\Big).
$$

**AEO-9: Meta-Differentiability.** All operators are smooth ⇒ end-to-end training of $W_\cdot,\Gamma,C,\tau$.

**AEO-10: Lyapunov Safety.** With small $\eta_a$ and AEO-5/6,

$$
\mathbb{E}\big[\mathcal{V}_{t+1}-\mathcal{V}_t\big]\le 0,\quad
\mathcal{V}_t=\text{LP}(\ell(\theta_t))+\tfrac{\gamma}{2}\|a_t-\bar a\|^2+\psi_\Omega(a_t).
$$

---

## 4) Key Propositions (informal)

**P1 — Normal-Form Equivalence.**
Any discrete-time optimizer with bounded internal state can be expressed in the AEO normal form by (i) folding internal adaptors into $a$ and $P_t$, and (ii) exposing its observable signals in $s_t$.
*Sketch:* Adam/RMS/Adagrad correspond to choices of $P_t(a,s)$ with stateful moments in $a$; LR schedulers are AEO with $W\!=\!0$ (open-loop special case).

**P2 — No-Free-Schedule.**
For drifting tasks, closed-loop $a_t$ (AEO-2/4/5) strictly dominates fixed schedules in regret/time-to-target under mild nonstationarity.
*Sketch:* Small-gain analysis with exogenous drift; feedback cancels first-order mismatch.

**P3 — Gauge Robustness.**
Effective-step control (AEO-8) yields invariance to gradient scale and batch size changes (up to second order).
*Sketch:* Treat gradient rescaling as time-reparameterization; $\gamma$ control compensates.

**P4 — Safety Under Energy Budget.**
With AEO-6 and Lipschitz $G_{\ell\!\gets\!a}$, the closed loop is input-to-state stable; blow-ups are prevented without manual LR cliffs.
*Sketch:* Lyapunov descent with budgeted multiplicative update.

---

## 5) Unifying Major ML Settings

* **Supervised / Self-Supervised:** AEO-SGDm or AEO-Adam: evolve $\alpha,\beta,\lambda,\sigma$ from $T,V,\rho,S$.
* **Reinforcement Learning:** rewards feed $R_t$; evolve entropy temperature, KL penalty, and step sizes; stabilize with $V_t$ and $\rho_t$.
* **Bayesian / Variational:** treat prior precision, posterior temperature, and KL weight as coefficients; AEO adapts them to maintain calibration and stability.
* **Diffusion / Generative:** adapt noise schedule weights, solver step sizes, and guidance scale with AEO-4/8 for improved FID vs. speed trade-offs.
* **Curricula & Augmentation:** treat augmentation strength, replay ratio, and sampling temperature as coefficients in $a_t$, evolved by the same laws.

---

## 6) Minimal Reference Algorithm (drop-in wrapper)

```python
# base trainer exposes: loss_t, reward_t (optional), grad_t, and step(θ, a)
# AEO state: a (coeff dict), σ (exploration), filters for T,V,R,ρ,S

s = measure_signals(loss_t, reward_t, grad_t)        # AEO-1
u = (Wp*tanh(-s.T/tau_p) - Wv*softplus(s.V/tau_v)    # AEO-2
     + Wr*tanh(s.R/tau_r) + Wc*(s.rho - rho_star)
     - Gamma*(a - a_bar) - grad_barrier(a))
u += C @ features(a)                                  # AEO-7
u += Wm*(s.rho - rho_star)                            # AEO-5

a = project(a * np.exp(eta_a * u))                    # AEO-3 (+ AEO-6 if used)
sigma *= np.exp(eta_sigma * noise_controller(s))      # AEO-4

θ = base_step(θ, a, sigma)                            # uses α,P(a,s), etc.
```

**Safe defaults:** tiny $\eta_a,\eta_\sigma\in[10^{-4},10^{-2}]$; $\rho^\star\in[0.2,0.4]$; multiplicative updates for positive coefficients; continuity budget on $a$.

---

## 7) Diagnostics & Tuning

* **Panel:** plot $T,V,\rho,S,\gamma$ with thresholds.
* **Divergence:** increase $W_v$, reduce $\eta_a$, tighten budget $B$.
* **Stall:** increase exploration terms in AEO-4; raise $W_p$.
* **Chatter:** increase $\Gamma$, smooth filters (larger $\tau$).

---

## 8) Empirical Predictions (falsifiable)

1. **Coupling:** best-performing regions satisfy a negative correlation between $\log\alpha$ and $\log \tfrac{1}{1-\beta}$ predicted by AEO-7.
2. **Gauge:** training remains stable under large batch/scale changes if $\gamma$ control is active.
3. **Nonstationarity:** under drifting data, AEO beats any fixed schedule in time-to-target at matched compute.
4. **Noise law:** when trend stalls but variance is low, learned $\sigma$ increases; when $V_t$ rises, $\sigma$ contracts.

---

## 9) Relationship to Prior Views

* **Not biology.** No reproduction/generations; “evolution” = continuous coefficient adaptation.
* **Beyond control alone.** Control stabilizes; **AEO** stabilizes **and** explores.
* **DSP-native.** Filtering, resonance, multiplicative gains, and noise shaping are first-class.
* **Meta-learnable.** All laws are differentiable—learn $W_\cdot,\Gamma,C,\tau$ end-to-end.

---

## 10) Slogan

**Machine Learning is Adaptive Evolution Optimization.**
Learning = **resonant feedback** on $\theta$; **evolution** = **closed-loop adaptation** of $a$.
Not survival of the fittest — **stability of the adaptive**.
