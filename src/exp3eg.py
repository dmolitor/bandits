import numpy as np
import pandas as pd
import plotnine as pn
from tqdm import tqdm
from typing import Callable, List, Optional

generator = np.random.default_rng(123)


def last(x: List):
    return x[len(x) - 1]


class Exp3EG:

    """
    This class implements the EXP3EG algorithm devised by Simchi-Levi and Wang:
    https://proceedings.mlr.press/v206/simchi-levi23a/simchi-levi23a.pdf.

    This algorithm casts the tradeoff between regret minimization and statistical
    power (ability to precisely estimate treatment effects between arms) as
    a Pareto optimal solution. The hyper-parameter `alpha` explicitly balances
    the trade-off between regret minimization and error (in treatment effect
    estimation) minimization.

    Attributes:
        ... Coming soon
    """

    def __init__(
        self,
        k: int,
        n: int,
        alpha: float,
        reward: Callable[[int], float],
        delta: Optional[float] = None,
        regret: Optional[Callable[[int], float]] = None
    ):

        # Initialize parameters
        self.reward = reward
        self.regret = regret
        self.arms = list(range(k))
        self.k = k
        self.n = n
        self.t = 0
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be ")
        self.alpha = alpha
        if delta is None:
            delta = 1/((2 * n)**2)
        self.delta = delta
        self.c = ((4 * (self.k**2) * (np.exp(2) + 1) + 2)**2) * (np.log(2/delta))**2

        # Initialize lists for sequential values
        self.probabilities = []
        self.rewards_hat = []
        self.regrets = []
        self.eliminated = []
        for _ in range(self.k):
            self.rewards_hat.append([0])
            self.probabilities.append([])
            self.eliminated.append([False])
        self.epsilon_t = [0]
        self.alpha_t = []
        self.selected_arm = []
    
    
    def fit(self, verbose: bool = True):

        """
        Fit the EXPE3EG algorithm over the given horizon.
        """

        if verbose:
            iter_seq = tqdm(range(self.n), total=self.n)
        else:
            iter_seq = range(self.n)
        for _ in iter_seq:
            self.pull()


    def pull(self):

        """
        Perform exactly one iteration of the EXPE3EG algorithm.
        """
        
        self.t += 1

        # Calculate hyper-parameters
        epsilon_t = 1/np.sqrt(self.c * self.t)
        alpha_t = 1/(self.k * (self.t**self.alpha))

        # Calculate arm selection probabilities
        eliminated = [last(x) for x in self.eliminated]
        a_t_comp_cardinality = np.sum(eliminated)
        denominator = np.sum(
            [
                np.exp(last(self.epsilon_t) * last(self.rewards_hat[arm]))
                for arm, elim in zip(self.arms, eliminated)
                if not elim
            ]
        )
        probabilities = []
        for arm, elim in zip(self.arms, eliminated):
            if elim:
                probabilities.append(alpha_t)
                self.probabilities[arm].append(alpha_t)
            else:
                p = (
                    (1 - a_t_comp_cardinality * alpha_t)
                    * (
                        np.exp(last(self.epsilon_t) * last(self.rewards_hat[arm]))
                        / denominator
                    )
                )
                probabilities.append(p)
                self.probabilities[arm].append(p)

        # Select arm to draw from
        assert np.isclose(np.sum(probabilities), 1, atol=1e-6), f"Arm probabilities must sum to 1; {probabilities}"
        selected_arm = np.argmax(generator.multinomial(1, probabilities))
        self.selected_arm.append(selected_arm)

        # Observe reward
        reward = self.reward(selected_arm)
        for arm in self.arms:
            if arm == selected_arm:
                self.rewards_hat[arm].append(
                    last(self.rewards_hat[arm]) + reward/probabilities[arm]
                )
            else:
                self.rewards_hat[arm].append(last(self.rewards_hat[arm]))
        
        # Calculate regret (if applicable)
        if self.regret is not None:
            self.regrets.append(self.regret(selected_arm))
        
        # Update which (if any) arms are eliminated
        max_reward = np.max([last(x) for x in self.rewards_hat])
        for arm in self.arms:
            if max_reward - last(self.rewards_hat[arm]) > 2 * np.sqrt(self.c * self.t):
                self.eliminated[arm].append(True)
            else:
                self.eliminated[arm].append(False)

        # Record values
        self.epsilon_t.append(epsilon_t)
        self.alpha_t.append(alpha_t)


if __name__ == "__main__":

    # Experiment
    def reward_bernoulli(arm: int):
        means = [0.9, 0.2]
        reward = generator.binomial(n=1, p=means[arm])
        return reward

    def regret_bernoulli(arm: int):
        means = [0.9, 0.2]
        regret = 0.9 - means[arm]
        return regret

    ### Does regret go down with changes in alpha??

    alpha_dict = {"regret": [], "ate": [], "alpha": []}

    for alpha in tqdm(range(0, 1001), total=1001):
        alpha = alpha/1000
        e3eg = Exp3EG(
            k=2,
            n=int(1e4),
            alpha=alpha,
            reward=reward_bernoulli,
            regret=regret_bernoulli
        )
        e3eg.fit(verbose=False)
        alpha_dict["regret"].append(np.sum(e3eg.regrets))
        alpha_dict["ate"].append(
            (last(e3eg.rewards_hat[0]) - last(e3eg.rewards_hat[1]))/e3eg.t
        )
        alpha_dict["alpha"].append(alpha)

    # Plot stuff
    alpha_df = pd.DataFrame(alpha_dict)
    alpha_df = pd.melt(
        alpha_df,
        id_vars=["alpha"],
        var_name="key",
        value_name="value"
    )

    (
        pn.ggplot(alpha_df, pn.aes(x="alpha", y="value"))
        + pn.geom_line()
        + pn.facet_wrap("~ key", nrow=2, scales="free")
        + pn.theme_538()
    )

    ### Does the rate of regret decrease quickly with n?

    n_dict = {"regret": [], "n": []}

    for n in tqdm(range(1000, 100000, 1000), total=100):
        e3eg = Exp3EG(
            k=2,
            n=n,
            alpha=0.5,
            reward=reward_bernoulli,
            regret=regret_bernoulli
        )
        e3eg.fit(verbose=False)
        n_dict["regret"].append(np.sum(e3eg.regrets))
        n_dict["n"].append(n)

    # Plot stuff
    n_df = pd.DataFrame(n_dict)

    (
        pn.ggplot(n_df, pn.aes(x="n", y="regret", group=1))
        + pn.geom_line()
        + pn.theme_538()
    )

    # print(
    #     f"Arms: {e3eg.arms}"
    #     + f"\nEliminated: {[last(x) for x in e3eg.eliminated]}"
    #     + f"\nProbabilities: {[last(x) for x in e3eg.probabilities]}"
    #     + f"\nNumber of rounds: {e3eg.t}"
    #     + f"\nReward estimates: {[last(x) for x in e3eg.rewards_hat]}"
    #     + f"\nSelected arms: {e3eg.selected_arm[994:999]}"
    #     + f"\nCumulative regret: {np.sum(e3eg.regrets)}"
    # )
    # ate = (last(e3eg.rewards_hat[0]) - last(e3eg.rewards_hat[1]))/e3eg.t