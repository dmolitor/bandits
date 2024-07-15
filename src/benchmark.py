import numpy as np
from src.lin_ucb import LinUCBDisjoint
from src.random import RandomPolicy
from src.yahoo_data_parser import parse_line, YahooDataParser
from tqdm import tqdm

generator = np.random.default_rng()

def last(x):
    return x[len(x) - 1]

def benchmark_lin_ucb(benchmarker, alpha: float):
    # Initial arms
    initial_arms = benchmarker.data_parser.next_record()
    initial_arms_records = (
        initial_arms.article_data
        [~initial_arms.article_data["article_feature_1"].isna()]
        [benchmarker.article_features]
        .to_records(index=False)
    )


    # Initialize bandit instance
    lin_ucb_dj = LinUCBDisjoint(
        alpha=alpha,
        horizon=benchmarker.data_parser.n_records,
        arms=initial_arms_records,
        arms_index="article_id"
    )

    # Record the pool of all arms
    all_arms = initial_arms_records["article_id"]
    cur_arms = initial_arms_records["article_id"]

    # Account for rewards and iterations
    cum_rewards = [0]
    aligned_rewards = [0]
    counter = 0

    for _ in tqdm(range(benchmarker.init_record_index, benchmarker.final_record_index - 1), total=benchmarker.final_record_index - benchmarker.init_record_index - 1):
        
        ## Update/deactivate arms as necessary

        # Get the next set of data
        new_data = benchmarker.data_parser.next_record()
        article_data = (
            new_data
            .article_data[~new_data.article_data["article_feature_1"].isna()]
            [benchmarker.article_features]
        )
        new_arms_records = article_data.to_records(index=False)
        cur_arms = new_arms_records["article_id"]

        # Handle new arms
        new_arms = new_arms_records[~np.isin(cur_arms, all_arms)]
        for arm in new_arms:
            lin_ucb_dj._add_arm(arm)
        
        # Handle inactive arms
        active_arms = lin_ucb_dj._active_arms()
        old_arms = active_arms[~np.isin(active_arms["article_id"], cur_arms)]
        for arm in old_arms:
            lin_ucb_dj._deactivate_arm(index_val=arm["article_id"])
        
        # Re-activate any arms if necessary
        inactive_arms = lin_ucb_dj._inactive_arms()
        dead_arms = inactive_arms[np.isin(inactive_arms["article_id"], cur_arms)]
        for arm in dead_arms:
            lin_ucb_dj._activate_arm(index_val=arm["article_id"])
        
        # Update all_arms
        all_arms = np.append(all_arms, new_arms["article_id"])

        ## Get user-specific data
        user_data = new_data.user_data
        clicked = user_data["user_click"][0]
        true_article_index = user_data["displayed_article_id"][0]
        user_record = (
            user_data
            .drop(
                labels=["timestamp", "displayed_article_id", "user_click"],
                axis=1
            )
            .to_records(index=False)
        )[0]

        # Define reward function
        def reward(index):
            return clicked

        # Run one iteration of the disjoint LinUCB algorithm
        selected_arm = lin_ucb_dj.pull(
            user_features=user_record,
            reward=reward,
            logging_index_val=true_article_index
        )

        if selected_arm == true_article_index:
            counter += 1
            cum_rewards.append(lin_ucb_dj.rewards.sum())
            aligned_rewards.append(lin_ucb_dj.rewards.sum()/counter)
    
    benchmarker.lin_ucb_disjoint_ctr = {"cum_rewards": cum_rewards, "aligned_rewards": aligned_rewards}
    benchmarker.reset()

class YahooDataBenchmark:
    """A class for benchmarking different bandit algorithms"""
    def __init__(self, path: str, n: float = np.inf):
        self.data_parser = YahooDataParser(path)
        if np.isinf(n):
            n = self.data_parser.n_records - 1
        final_record_index = generator.integers(n, self.data_parser.n_records)
        init_record_index = final_record_index - n
        self.data_parser.current_record = init_record_index
        self.init_record_index = init_record_index
        self.final_record_index = final_record_index
        self.article_features = ["article_id"] + [f"article_feature_{x}" for x in range(1, 7)]
    
    def random_policy_ctr(self):
        rewards = {"cum_rewards": [0], "aligned_rewards": [0]}
        counter = 0
        for i in tqdm(range(self.init_record_index, self.final_record_index, 1), total=self.final_record_index - self.init_record_index):
            counter += 1
            reward = parse_line(self.data_parser.read_line(i))["user_click"]
            rewards["cum_rewards"].append((last(rewards["cum_rewards"]) + reward))
            rewards["aligned_rewards"].append((last(rewards["cum_rewards"]) / counter))
        self.true_ctr = rewards
    
    def reset(self):
        self.data_parser.current_record = self.init_record_index
        self.true_ctr = None
        self.lin_ucb_ctr = None
    
    def benchmark(self, alpha: float):
        benchmark_lin_ucb(self, alpha=alpha)

# Initialize data parser
data_path = "/Users/dmolitor/Downloads/R6/ydata-fp-td-clicks-v1_0.20090501.gz"
benchmarker = YahooDataBenchmark(path=data_path, n=np.inf)
benchmarker.random_policy_ctr()
random_ctr = last(benchmarker.true_ctr["aligned_rewards"])

# Benchmark
vals = {"alpha": [], "scaled_ctr": [], "ctr": []}
for alpha in np.arange(0, 1.1, 0.1):
    print(f"Alpha: {alpha}")
    vals["alpha"].append(alpha)
    benchmarker.benchmark(alpha=alpha)
    ctr = last(benchmarker.lin_ucb_disjoint_ctr["aligned_rewards"])
    vals["scaled_ctr"].append(ctr/random_ctr)
    vals["ctr"].append(ctr)
    benchmarker.reset()
