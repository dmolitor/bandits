from mpire import WorkerPool
import numpy as np
import pandas as pd
from pathlib import Path
import plotnine as pn
from tqdm import tqdm

generator = np.random.default_rng(seed=123)
base_dir = Path(__file__).resolve().parent.parent

def which_max(a: np.ndarray):
    return np.equal(a, np.max(a))

class BernoulliBanditTS:
    """A Bernoulli Bandit estimated via Thompson Sampling"""

    def __init__(self, means, n, beta_params):
        self.beta_params = beta_params
        self.means = means
        self.n = n
        self.optimal_arm = means.index(max(means))
        self.probabilities = []
        self.regret = []
        self.rewards = []
        self.selected_arm = []
        self.selected_arm_80 = None
        for _ in means:
            self.probabilities.append([])
            self.regret.append([])
            self.rewards.append([])
    
    def best_arm(self):
        return np.argmax(self.sample())
    
    def fit(self, warmup_n = 0, verbose = True):
        if warmup_n > 0:
            if verbose:
                print("Warmp-up phase ...")
                n_warmup = tqdm(range(warmup_n), total=warmup_n)
            else:
                n_warmup = range(warmup_n)
            for iter in n_warmup:
                self.pull(iter)
        if verbose:
            print("Adaptive phase ...")
            n_remaining = tqdm(range(self.n), total=self.n)
        else:
            n_remaining = range(self.n)
        for _ in n_remaining:
            self.pull()
    
    def pull(self, selected_arm = None):
        """Sample from the bandit arm posteriors and 'pull' one of them"""
        if selected_arm is None:
            optimal_arm_estimates = self.sample()
            for idx in range(len(self.means)):
                self.probabilities[idx].append(optimal_arm_estimates[idx])
            selected_arm = np.argmax(optimal_arm_estimates)
            if np.any(optimal_arm_estimates >= 0.8) and self.selected_arm_80 is None:
                self.selected_arm_80 = selected_arm
        else:
            # This allows me to pass in a number greater than the number of
            # arms and it will map it directly to one of the arms.
            selected_arm = selected_arm % len(self.means)
            for idx in range(len(self.means)):
                if idx == selected_arm:
                    self.probabilities[idx].append(1)
                else:
                    self.probabilities[idx].append(0)
        self.selected_arm.append(selected_arm)
        selected_arm_mean = self.means[selected_arm]
        reward = generator.binomial(1, selected_arm_mean)
        self.rewards[selected_arm].append(reward)
        if reward == 1:
            self.beta_params["a"][selected_arm] += 1
        else:
            self.beta_params["b"][selected_arm] += 1
        regret = self.means[self.optimal_arm] - self.means[selected_arm]
        self.regret[selected_arm].append(regret)
        self.n -= 1
    
    def sample(self):
        """
        Thompson sampling. Draw 1e3 draws from each arm's posterior
        distribution and calculate the probability of it being the
        optimal arm.
        """
        samples = np.random.beta(
            self.beta_params["a"],
            self.beta_params["b"],
            size=(int(1e3), len(self.means))
        )
        samples_max = samples == samples.max(axis=1, keepdims=True)
        samples_proportion = samples_max.mean(axis=0)
        return samples_proportion

if __name__ == "__main__":

    ### Run simulations
    results_dict = {
        "id": [],
        "param_value": [],
        "expected_regret": [],
        "correct_arm": [],
        "correct_arm_80": [],
        "min_opt_gap": [],
        "opt_mean": [],
        "warmup": []
    }

    id = 0

    for param_value in range(1, 111, 10):

        for warmup in [0, 1000, 2000]:

            for best_mean in [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55]:

                # Check if done already
                id += 1
                if id in results_dict["id"]:
                    continue
                
                def fit_sim(_):
                    bandit = BernoulliBanditTS(
                        means=[best_mean] + [0.5]*15,
                        n=5000,
                        beta_params={"a": [param_value]*16, "b": [param_value]*16}
                    )
                    if warmup > 0:
                        bandit.fit(warmup_n=warmup, verbose=False)
                    else:
                        bandit.fit(verbose=False)
                    regret = np.sum([np.sum(x) for x in bandit.regret])
                    correct_arm = int(bandit.best_arm() == 0)
                    correct_arm_80 = int(bandit.selected_arm_80 == 0)
                    return (regret, correct_arm, correct_arm_80)

                print(
                    f"Id: {id}; True mean: {best_mean}; "
                    + f"Warmup N: {warmup}; ~Beta({param_value}, {param_value})"
                )
                with WorkerPool(n_jobs=15) as pool:
                    results = pool.map(fit_sim, range(500), progress_bar=True)
                regrets = [x[0] for x in results]
                correct_arms = [x[1] for x in results]
                correct_arms_80 = [x[2] for x in results]
                results_dict["id"].append(id)
                results_dict["param_value"].append(param_value)
                results_dict["expected_regret"].append(np.mean(regrets))
                results_dict["correct_arm"].append(np.mean(correct_arms))
                results_dict["correct_arm_80"].append(np.mean(correct_arms_80))
                results_dict["opt_mean"].append(best_mean)
                results_dict["min_opt_gap"].append(best_mean - 0.5)
                results_dict["warmup"].append(warmup)

    ### Collect simulation results into a dataframe
    results = pd.DataFrame(results_dict)
    results["warmup_label"] = results["warmup"].apply(lambda x: f"Warmup N: {x}")


    ### Plot it!
    regret_plot = (
        pn.ggplot(
            results,
            pn.aes(
                x="min_opt_gap",
                y="expected_regret",
                color="factor(param_value)",
                group="param_value"
            )
        )
        + pn.geom_line()
        + pn.facet_wrap("~ warmup_label", nrow=3, scales="fixed")
        + pn.labs(
            x="Optimality gap",
            y="Mean expected regret",
            title="Bernoulli Bandit; k=16, n=5,000",
            color="Beta"
        )
        + pn.theme_538()
        + pn.theme(
            axis_title=pn.element_text(weight="bold"),
            plot_title=pn.element_text(weight="bold", hjust=0.5)
        )
    )
    pn.ggsave(
        regret_plot,
        filename=base_dir/"figures"/"ts_expected_regret.png",
        width=8,
        height=6,
        dpi=300
    )

    correct_plot = (
        pn.ggplot(
            results,
            pn.aes(
                x="min_opt_gap",
                y="correct_arm",
                color="factor(param_value)",
                group="param_value"
            )
        )
        + pn.geom_line()
        + pn.facet_wrap("~ warmup_label", nrow=3)
        + pn.labs(
            x="Optimality gap",
            y="Fraction selecting correct arm",
            title="Bernoulli Bandit; k=16, n=5,000",
            color="Beta"
        )
        + pn.theme_538()
        + pn.theme(
            axis_title=pn.element_text(weight="bold"),
            plot_title=pn.element_text(weight="bold", hjust=0.5)
        )
    )
    pn.ggsave(
        correct_plot,
        filename=base_dir/"figures"/"ts_correct_arm.png",
        width=8,
        height=6,
        dpi=300
    )

    correct_plot_80 = (
        pn.ggplot(
            results,
            pn.aes(
                x="min_opt_gap",
                y="correct_arm_80",
                color="factor(param_value)",
                group="param_value"
            )
        )
        + pn.geom_line()
        + pn.facet_wrap("~ warmup_label", nrow=3)
        + pn.labs(
            x="Optimality gap",
            y="Fraction selecting correct arm (80% cutoff)",
            title="Bernoulli Bandit; k=16, n=5,000",
            color="Beta"
        )
        + pn.theme_538()
        + pn.theme(
            axis_title=pn.element_text(weight="bold"),
            plot_title=pn.element_text(weight="bold", hjust=0.5)
        )
    )
    pn.ggsave(
        correct_plot_80,
        filename=base_dir/"figures"/"ts_correct_arm_80.png",
        width=8,
        height=6,
        dpi=300
    )

    results.to_csv(base_dir/"data"/"ts_results.csv", index=False)