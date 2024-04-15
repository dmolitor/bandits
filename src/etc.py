import numpy as np
import pandas as pd
from pathlib import Path
import plotnine as pn
from tqdm import tqdm

generator = np.random.default_rng(seed=123)
base_dir = Path(__file__).resolve().parent.parent
# For interactive use, uncomment below
# base_dir = Path().resolve()

# Gaussian bandit
class GaussianBandit:
    def __init__(self, mean: int, n: int):
        self.delta = mean
        self.means = [0, -mean]
        self.n = n
        self.n_remaining = n
        self.realizations = [[], []]
        self.regret = [[], []]

    def fit(self, m: None | int = None) -> None:
        """Play the ETC algorithm with a Gaussian bandit"""
        if m is None:
            m = self.optimize_m()
        for _ in range(m):
            for arm in [0, 1]:
                self.pull(arm)
        mean_arm_rewards = [np.mean(rewards) for rewards in self.realizations]
        optimal_arm = int(np.argmax(mean_arm_rewards))
        self.optimal_arm = optimal_arm
        for _ in range(self.n_remaining):
            self.pull(optimal_arm)
        return None
    
    def k(self) -> int:
        """Return the number of arms"""
        return len(self.means)
    
    def optimize_m(self) -> int:
        """Find the optimal m for a 1-subgaussian bandit under ETC"""
        optimal_m = max(
            int(np.ceil((4/(self.delta**2))*np.log((self.n*self.delta**2)/4))),
            1
        )
        return optimal_m
    
    def pull(self, arm: int) -> None:
        """Pull a bandit arm"""
        reward = generator.normal(self.means[arm], 1, size=1)[0]
        self.realizations[arm].append(reward)
        self.regret[arm].append(0 - self.means[arm])
        self.n_remaining = int(self.n_remaining - 1)
        return None
    
    def regret_cumulative(self) -> float:
        """Cumulative regret over `self.n` rounds"""
        return sum(self.regret[0] + self.regret[1])
    
    def regret_ub(self) -> float:
        """Cumulative regret upper bound over `self.n` rounds"""
        rhs = (
            self.delta
            + (4/self.delta)*(1 + np.max([0, np.log(self.n*self.delta**2/4)]))
        )
        lhs = self.n*self.delta
        ub = np.min([lhs, rhs])
        return ub

# Experiment 6.1
if __name__ == "__main__":
    
    # 6.9 b)
    # Initialize a Gaussian bandit with k = 2 and means mu_1 = 0 and mu_2 = -d
    # where d is chosen s.t. 0 <= d <= 1. Also, by assumption, all bandit
    # arms are 1-subgaussian which implies that the variance of each arm
    # is 1. Evaluate the mean cumulative regret and upper regret bounds
    # for 10,000 simulations on each value of d. Then plot the cumulative
    # regret and regret upper bound curves.
    regret_and_ub = []
    deltas = [x/40 for x in range(1, 40)]
    for delta in deltas:
        print(f"Delta: {delta}")
        regrets_cumulative = []
        regrets_ub = []
        for _ in tqdm(range(int(1e5)), total=int(1e5)):
            bandit = GaussianBandit(mean=delta, n=1000)
            bandit.fit()
            regrets_cumulative.append(bandit.regret_cumulative())
            regrets_ub.append(bandit.regret_ub())
        regret_and_ub.append(
            {
                "regret_cumulative": np.mean(regrets_cumulative),
                "regret_ub": np.mean(regrets_ub)
            }
        )
    regret_df = (
        pd
        .melt(
            (
                pd.DataFrame(regret_and_ub)
                .assign(delta=deltas)
            ),
            id_vars=["delta"],
            var_name="type",
            value_name="value"
        )
        .assign(type=lambda x: x["type"].map(lambda s: "Cumulative Regret" if s == "regret_cumulative" else ("Regret Upper Bound" if s == "regret_ub" else s)))
    )

    regret_plot = (
        pn.ggplot(data = regret_df)
        + pn.geom_line(pn.aes(x="delta", y="value", linetype="type"))
        + pn.labs(x="delta", y="Expected regret", linetype="")
        + pn.theme_538()
    )
    pn.ggsave(
        regret_plot,
        filename=base_dir/"figures"/"6_9b.png",
        width=8,
        height=6,
        dpi=300
    )

    # 6.9 c and d)
    # Fix delta = 1/10 and plot the expected regret as a function of m with
    # n = 2000.
    ms = range(10, 410, 10)
    regrets = []
    sds = []
    for m in ms:
        print(f"m: {m}")
        regrets_cumulative = []
        for _ in tqdm(range(int(1e5)), total=int(1e5)):
            bandit = GaussianBandit(mean=1/10, n=2000)
            bandit.fit(m=m)
            regrets_cumulative.append(bandit.regret_cumulative())
        regrets.append(
            {
                "regret_cumulative": np.mean(regrets_cumulative),
                "regret_sd": np.std(regrets_cumulative),
                "m": m}
        )
    regret_df = pd.DataFrame(regrets)

    # 6.9c
    regret_plot = (
        pn.ggplot(data = regret_df)
        + pn.geom_line(pn.aes(x="m", y="regret_cumulative"))
        + pn.labs(y="Expected regret")
        + pn.theme_538()
    )
    pn.ggsave(
        regret_plot,
        filename=base_dir/"figures"/"6_9c.png",
        width=8,
        height=6,
        dpi=300
    )

    # 6.9 d
    regret_sd_plot = (
        pn.ggplot(data = regret_df)
        + pn.geom_line(pn.aes(x="m", y="regret_sd"))
        + pn.labs(y="Standard deviation of the regret")
        + pn.theme_538()
    )
    pn.ggsave(
        regret_sd_plot,
        filename=base_dir/"figures"/"6_9d.png",
        width=8,
        height=6,
        dpi=300
    )