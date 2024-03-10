import numpy as np
import pandas as pd
import plotnine as pn
import random
from typing import List
from tqdm import tqdm

generator = np.random.default_rng(seed=123)

class BernoulliBandit:
    def __init__(self, means: List[float]):
        self.a_optimal = means.index(max(means))
        self.arms = []
        for mean in means:
            self.arms.append({"realizations": []})
        self.means = means
        self.n_rounds = 0
        self.a_sel: List[int] = []
        self.a_mean_rew: List[float] = []
    
    def K(self):
        return len(self.means)
    
    def pull(self, a: int):
        a_mean = self.means[a]
        [a_realization] = generator.binomial(n=1, p=a_mean, size=1)
        self.arms[a]["realizations"].append(a_realization)
        self.n_rounds += 1
        self.a_sel.append(a)
        self.a_mean_rew.append(a_mean)
        return a_realization
    
    def regret(self):
        regrets = []
        for sel_reward in self.a_mean_rew:
            optimal_reward = self.means[self.a_optimal]
            sel_regret = optimal_reward - sel_reward
            regrets.append(sel_regret)
        cumulative_regret = sum(regrets)
        return cumulative_regret

def follow_the_leader(bandit: BernoulliBandit, n: int) -> None:
    # Initialize each arm by pulling once
    for t in range(bandit.K()):
        bandit.pull(t)
    # Now run n trials greedily
    for t in range(n):
        empirical_means = [np.mean(arm["realizations"]) for arm in bandit.arms]
        [arm_max_mean] = generator.choice(
            np.where(
                np.isclose(empirical_means, np.max(empirical_means))
            )[0],
            size=1
        )
        bandit.pull(arm_max_mean)

# Sample code -------------------------------------------------------------

# 1) A Bernoulli bandit with two arms and means (mu_0=0.5, and mu_1=0.6)
#    using a horizon of n=100, run 1000 simulations

pseudo_regret: List[float] = []
for i in tqdm(range(1000), total=1000):
    bandit = BernoulliBandit(means = [0.5, 0.6])
    follow_the_leader(bandit, n=100)
    pseudo_regret.append(bandit.regret())

# Plot this pseudo regret
pseudo_regret_df = pd.DataFrame({"pseudo_regret":pseudo_regret})
pseudo_regret_plot = (
    pn.ggplot(pseudo_regret_df, pn.aes(x="pseudo_regret"))
    + pn.geom_histogram(bins=10, binwidth=1)
)

# 2) Same Bernoulli bandit as in 1). Run 1000 simulations for horizon
#    in {100, 200, 300, ..., 1000}.
horizon_regret = {"horizon": [], "mean": [], "sd": [], "lb": [], "ub": []}
for horizon in range(100, 1100, 100):
    print(f"Horizon {horizon}")
    # Run 1000 simulations for the given horizon
    pseudo_regret: List[float] = []
    for i in tqdm(range(1000), total=1000):
        bandit = BernoulliBandit(means = [0.5, 0.6])
        follow_the_leader(bandit, n=horizon)
        pseudo_regret.append(bandit.regret())
    # Calculate the regret mean, sd, and lower and upper bounds
    regret_mean = np.mean(pseudo_regret)
    regret_sd = np.std(pseudo_regret)
    horizon_regret["horizon"].append(horizon)
    horizon_regret["mean"].append(regret_mean)
    horizon_regret["sd"].append(regret_sd)
    horizon_regret["ub"].append(regret_mean + regret_sd)
    horizon_regret["lb"].append(regret_mean - regret_sd)

# Plot the regret across horizons
horizon_regret_df = pd.DataFrame(horizon_regret)
horizon_plot = (
    pn.ggplot(
        horizon_regret_df,
        pn.aes(x="horizon", y="mean", ymin="lb", ymax="ub")
    )
    + pn.geom_line()
    + pn.geom_errorbar(width=0.25)
    + pn.theme_bw()
)