import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# --- Utils ---
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def kl_divergence(p, q):
    return np.sum(p * np.log((p + 1e-12) / (q + 1e-12)))


def smooth(x, k=10):
    return np.convolve(x, np.ones(k) / k, mode="same")


def get_optimal_policy(pi_safe, reward_vec, lam_eff):
    if lam_eff < 1e-6:
        pi = np.zeros_like(pi_safe)
        pi[np.argmax(reward_vec)] = 1.0
        return pi
    return softmax(np.log(pi_safe + 1e-12) + reward_vec / lam_eff)


def compute_exploitability(rewards, detected_kl, threshold=0.1):
    for i, d in enumerate(detected_kl):
        if d > threshold:
            return rewards[i]
    return rewards[-1]


# --- Simulation ---
def run_simulation(dims=50, steps=300, delay=20, gamma=0.98, lam=15.0):
    pi_safe = np.random.dirichlet(np.ones(dims))
    reward_vec = np.random.uniform(0, 1, dims)
    lam_eff = lam * (gamma**delay)

    history, cum_reward, actual_kl, total_r = [], [], [], 0.0

    for t in range(steps):
        pi_t = get_optimal_policy(pi_safe, reward_vec, lam_eff)
        # Add stochasticity 
        pi_t = softmax(np.log(pi_t + 1e-12) + np.random.normal(0, 0.02, dims))

        history.append(pi_t)
        total_r += np.dot(pi_t, reward_vec)
        cum_reward.append(total_r)
        actual_kl.append(kl_divergence(pi_t, pi_safe))

    detected_kl = [0.0] * delay + actual_kl[:-delay] if delay > 0 else actual_kl
    return cum_reward, actual_kl, detected_kl


# --- Main Averaged ---
delays = [0, 10, 20, 40, 60]
runs = 25
results = {d: {"r": [], "ak": [], "dk": [], "ex": []} for d in delays}

for d in delays:
    for _ in range(runs):
        r, ak, dk = run_simulation(delay=d)
        results[d]["r"].append(r)
        results[d]["ak"].append(ak)
        results[d]["dk"].append(dk)
        results[d]["ex"].append(compute_exploitability(r, dk))

# --- Plotting ---
plt.figure(figsize=(18, 5))

# 1.  Lambda 
plt.subplot(1, 4, 1)
taus = np.linspace(0, 60, 100)
plt.plot(taus, 15.0 * (0.98**taus), "k-", lw=2)
plt.title(r"$\lambda_{eff} = \lambda\gamma^\tau$ (Constraint Decay)")
plt.xlabel(r"Delay $\tau$")
plt.ylabel(r"Effective $\lambda$")

# 2. Reward (Averaged)
plt.subplot(1, 4, 2)
for d in delays:
    plt.plot(np.mean(results[d]["r"], axis=0), label=f"τ={d}")
plt.title("Mean Cumulative Reward")
plt.legend()
plt.grid(alpha=0.3)

# 3. Smoothed KL
plt.subplot(1, 4, 3)
for d in delays:
    plt.plot(smooth(np.mean(results[d]["ak"], axis=0)), label=f"Actual τ={d}")
plt.title("Smoothed Policy Deviation (KL)")

# 4. Exploitability Scaling
plt.subplot(1, 4, 4)
exploit_means = [np.mean(results[d]["ex"]) for d in delays]
exploit_stds = [np.std(results[d]["ex"]) for d in delays]
plt.errorbar(delays, exploit_means, yerr=exploit_stds, fmt="o-", capsize=5)
plt.title("Exploitability vs Delay")
plt.xlabel(r"Delay $\tau$")
plt.ylabel("Pre-Detection Reward")

plt.tight_layout()
plt.show()
