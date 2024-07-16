import numpy as np
import pandas as pd
from pathlib import Path
import plotnine as pn
from tqdm import tqdm
from typing import List

generator = np.random.default_rng(seed=123)
base_dir = Path(__file__).resolve().parent.parent
# For interactive use, uncomment below
# base_dir = Path().resolve()


def last(x: List):
    return x[len(x) - 1]


# Adversarial Exp3 algorithm
class AdversarialExp3:

    def __init__(self, rewards: List[float], n: int, eta: float):
        self.s_hat = [[0], [0]]
        self.probabilities = [[], []]
        self.rewards = rewards
        self.realizations = []
        self.k = len(rewards)
        self.n = n
        self.eta = eta

    def fit(self) -> None:
        """Play the Exp3 algorithm"""
        for t in range(self.n):
            p_t_denominator = np.sum([np.exp(self.eta * last(x)) for x in self.s_hat])
            p_t = [np.exp(self.eta * last(x)) / p_t_denominator for x in self.s_hat]
            a_t = np.argmax(generator.multinomial(1, p_t))
            reward = generator.binomial(1, self.rewards[a_t])
            self.realizations.append(reward)
            for idx in range(self.k):
                self.probabilities[idx].append(p_t[idx])
                s_ti = (
                    last(self.s_hat[idx])
                    + 1
                    - (int(a_t == idx) * (1 - reward)) / p_t[idx]
                )
                self.s_hat[idx].append(s_ti)

    def regret_cumulative(self) -> List[float]:
        cum_regret = [
            (self.n * reward) - np.sum(self.realizations) for reward in self.rewards
        ]
        return np.max(cum_regret)


# Exercises
if __name__ == "__main__":

    # 11.9 Stochastic Bernoulli Bandit with Exp3
    # Plot loose regret as a function of learning rate with fixed horizon T=1e4.
    # Average loose regret across 100 replications at each eta value.
    regret_dict = {"eta": [], "regret": []}
    etas = range(1, 1010, 10)
    while True:
        try:
            for e in tqdm(etas, total=len(etas)):
                e = e / 10000
                if e in regret_dict["eta"]:
                    continue
                regrets = []
                for m in range(100):
                    adv_bandit = AdversarialExp3(rewards=[0.5, 0.55], n=int(1e4), eta=e)
                    adv_bandit.fit()
                    regrets.append(adv_bandit.regret_cumulative())
                regret_dict["eta"].append(e)
                regret_dict["regret"].append(np.mean(regrets))

            break
        except:
            continue

    # Plot regret
    regret_df = pd.DataFrame(regret_dict)
    regret_plot = (
        pn.ggplot(regret_df, pn.aes(x="eta", y="regret", group=1))
        + pn.geom_line()
        + pn.labs(
            x="eta", y="Expected regret", title="Regret as a function of learning rate"
        )
        + pn.theme_538()
        + pn.theme(
            axis_title=pn.element_text(weight="bold"),
            plot_title=pn.element_text(weight="bold", hjust=0.5),
        )
    )
    pn.ggsave(
        regret_plot,
        filename=base_dir / "figures" / "11_9.png",
        width=8,
        height=6,
        dpi=300,
    )
