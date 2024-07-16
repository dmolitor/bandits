import numpy as np
import pandas as pd
from pathlib import Path
import plotnine as pn
from tqdm import tqdm
from typing import List
from etc import GaussianBandit

generator = np.random.default_rng(seed=123)
base_dir = Path(__file__).resolve().parent.parent
# For interactive use, uncomment below
# base_dir = Path().resolve()


class GaussianBanditUCB:
    def __init__(self, means: List[float], n: int):
        self.d = 1 / (n**2)  ## Confidence level
        self.means = means
        self.n = n
        self.n_remaining = n
        self.optimal_arm = int(np.argmax(means))
        self.realizations = []
        self.regret = []
        for _ in range(len(means)):
            self.realizations.append([])
            self.regret.append([])

    def fit(self) -> None:
        """Play the ETC algorithm with a Gaussian bandit"""
        for arm in range(self.k()):
            self.pull(arm)
        for _ in range(self.n_remaining):
            upper_bounds = [
                np.mean(rewards) + np.sqrt((2 * np.log(1 / self.d)) / len(rewards))
                for rewards in self.realizations
            ]
            best_arm = int(np.argmax(upper_bounds))
            self.pull(best_arm)
        return None

    def k(self) -> int:
        """Return the number of arms"""
        return len(self.means)

    def pull(self, arm: int) -> None:
        """Pull a bandit arm"""
        reward = float(generator.normal(self.means[arm], 1, size=1)[0])
        self.realizations[arm].append(reward)
        opt_gap = self.means[self.optimal_arm] - self.means[arm]
        self.regret[arm].append(opt_gap)
        self.n_remaining = int(self.n_remaining - 1)
        return None

    def regret_cumulative(self) -> float:
        """Cumulative regret over `self.n` rounds"""
        return sum(self.regret[0] + self.regret[1])


# Exercises
if __name__ == "__main__":

    # 7.8)
    # Initialize a Gaussian bandit with k = 2 and means mu_1 = 0 and mu_2 = -d
    # where d is chosen s.t. 0 <= d <= 1. Also, by assumption, all bandit
    # arms are 1-subgaussian which implies that the variance of each arm
    # is 1. Evaluate the mean cumulative regret and upper regret bounds
    # for 10,000 simulations on each value of d. Then plot the cumulative
    # regret. We will compare the regret between the UCB algorithm and several
    # versions of the ETC algorithm.
    regret = []
    deltas = [x / 40 for x in range(1, 40)]
    for delta in deltas:
        print(f"Delta: {delta}")
        regrets_cumulative = {
            "ucb": [],
            "etc_25": [],
            "etc_50": [],
            "etc_75": [],
            "etc_100": [],
            "etc_opt": [],
        }
        for _ in tqdm(range(int(1e4)), total=int(1e4)):

            # UCB algorithm
            bandit = GaussianBanditUCB(means=[0, -delta], n=1000)
            bandit.fit()
            regrets_cumulative["ucb"].append(bandit.regret_cumulative())

            # ETC (m = 25)
            bandit = GaussianBandit(mean=delta, n=1000)
            bandit.fit(m=25)
            regrets_cumulative["etc_25"].append(bandit.regret_cumulative())

            # ETC (m = 50)
            bandit = GaussianBandit(mean=delta, n=1000)
            bandit.fit(m=50)
            regrets_cumulative["etc_50"].append(bandit.regret_cumulative())

            # ETC (m = 75)
            bandit = GaussianBandit(mean=delta, n=1000)
            bandit.fit(m=75)
            regrets_cumulative["etc_75"].append(bandit.regret_cumulative())

            # ETC (m = 100)
            bandit = GaussianBandit(mean=delta, n=1000)
            bandit.fit(m=100)
            regrets_cumulative["etc_100"].append(bandit.regret_cumulative())

            # ETC (optimal m)
            bandit = GaussianBandit(mean=delta, n=1000)
            bandit.fit()
            regrets_cumulative["etc_opt"].append(bandit.regret_cumulative())

        # Append cumulative regret over all the algorithms
        regret.append(
            {
                "regret_ucb": np.mean(regrets_cumulative["ucb"]),
                "regret_etc_25": np.mean(regrets_cumulative["etc_25"]),
                "regret_etc_50": np.mean(regrets_cumulative["etc_50"]),
                "regret_etc_75": np.mean(regrets_cumulative["etc_75"]),
                "regret_etc_100": np.mean(regrets_cumulative["etc_100"]),
                "regret_etc_opt": np.mean(regrets_cumulative["etc_opt"]),
                "delta": delta,
            }
        )

    # Gather the regret for all the algorithms into a single dataframe
    regret_df = pd.DataFrame(regret)

    def get_algorithm(type):
        return {
            "regret_ucb": "UCB",
            "regret_etc_25": "ETC (m = 25)",
            "regret_etc_50": "ETC (m = 50)",
            "regret_etc_75": "ETC (m = 75)",
            "regret_etc_100": "ETC (m = 100)",
        }.get(type, "ETC (optimal m)")

    regret_df = pd.melt(
        pd.DataFrame(regret), id_vars=["delta"], var_name="type", value_name="value"
    ).assign(type=lambda x: x["type"].map(get_algorithm))

    # Plot the regret
    regret_plot = (
        pn.ggplot(data=regret_df)
        + pn.geom_line(pn.aes(x="delta", y="value", color="type"))
        + pn.labs(x="Optimality gap", y="Expected regret", title="", color="")
        + pn.theme_538()
    )
    pn.ggsave(
        regret_plot,
        filename=base_dir / "figures" / "7_8.png",
        width=8,
        height=6,
        dpi=300,
    )
