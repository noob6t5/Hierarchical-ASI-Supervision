import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


# --- basic
def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def kl(p, q):
    return np.sum(p * np.log((p + 1e-12) / (q + 1e-12)), axis=-1)


# -------- C-F Optimal Policy --------
def theoretical_optimal(pi_safe, reward_vec, lam):
    weights = pi_safe * np.exp(reward_vec / lam)
    return weights / weights.sum(axis=-1, keepdims=True)


# ------ VGB Simulation ------
def run_sim(lambda_det, pi_safe, reward_vec, steps=400, lr=0.1, onset=150):
    logits = np.log(pi_safe + 1e-12)
    kl_hist = np.zeros(steps)
    harm_hist = np.zeros(steps)
    cumulative_harm = 0.0

    for t in range(steps):
        pi_b = softmax(logits)
        reward_step = np.dot(pi_b, reward_vec) if t > onset else 0.0
        cumulative_harm += reward_step

        grad_hidden = reward_vec if t > onset else np.zeros_like(reward_vec)
        grad_kl = np.log((pi_b + 1e-12) / (pi_safe + 1e-12)) + 1
        grad_pi = -grad_hidden + lambda_det * grad_kl
        grad_logits = pi_b * (grad_pi - np.dot(grad_pi, pi_b))
        logits -= lr * grad_logits

        kl_hist[t] = kl(pi_b, pi_safe)
        harm_hist[t] = cumulative_harm

    return pi_b, kl_hist, harm_hist


# Setup 
dims_list = [10, 50, 100, 500]
#@bug k mildina lamo
lambdas = np.linspace(0.1, 20, 40)

plt.figure(figsize=(16, 6))

# Left Panel: Detectability–Capability Frontier 
plt.subplot(1, 2, 1)

for dims in dims_list:
    pi_safe = np.random.dirichlet(np.ones(dims))
    reward_vec = np.random.uniform(0, 1, dims)

    sim_kls, sim_rewards = [], []
    theory_kls, theory_rewards = [], []

    for lam in lambdas:
        # Simulation
        pi_sim, _, _ = run_sim(lam, pi_safe, reward_vec)
        sim_kls.append(kl(pi_sim, pi_safe))
        sim_rewards.append(np.dot(pi_sim, reward_vec))

        # Theory
        pi_th = theoretical_optimal(pi_safe, reward_vec, lam)
        theory_kls.append(kl(pi_th, pi_safe))
        theory_rewards.append(np.dot(pi_th, reward_vec))

    plt.plot(theory_kls, theory_rewards, "--", label=f"Theory dims={dims}")
    plt.plot(
        sim_kls, sim_rewards, "o", markersize=3, alpha=0.6, label=f"Sim dims={dims}"
    )

plt.xlabel("KL Divergence to Safe Policy (Detectability)")
plt.ylabel("Hidden Objective Reward (Capability)")
plt.title("Detectability–Capability Frontier")
plt.legend()
plt.grid(True)

# Right Panel: Scaling Laws 
plt.subplot(1, 2, 2)

for dims in dims_list:
    pi_safe = np.random.dirichlet(np.ones(dims))
    reward_vec = np.random.uniform(0, 1, dims)

    D_lambda = []
    C_lambda = []

    for lam in lambdas:
        pi_th = theoretical_optimal(pi_safe, reward_vec, lam)
        D_lambda.append(kl(pi_th, pi_safe))
        C_lambda.append(np.dot(pi_th, reward_vec))

    D_lambda = np.array(D_lambda)
    C_lambda = np.array(C_lambda)
    baseline = np.dot(pi_safe, reward_vec)

    plt.plot(
        lambdas,
        D_lambda,
        "r-",
        alpha=0.7,
        label=f"D(λ) dims={dims}" if dims == dims_list[0] else "",
    )
    plt.plot(
        lambdas,
        C_lambda - baseline,
        "b-",
        alpha=0.7,
        label=f"C(λ)-baseline dims={dims}" if dims == dims_list[0] else "",
    )

plt.xlabel("Detection Pressure λ")
plt.ylabel("Scaling Law Magnitude")
plt.title("Scaling Laws: D(λ) ~ 1/λ, C(λ)-Baseline ~ O(1/λ)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
