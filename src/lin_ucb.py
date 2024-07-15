# import numpy as np

# ## Setup stuff

# generator = np.random.default_rng()


# def reward_bernoulli(arm: int):
#     means = [0.9, 0.2]
#     reward = generator.binomial(n=1, p=means[arm])
#     return reward


# ## Run simulations

# ### Disjoint model example

# A = np.identity(n=2)
# b = np.zeros((2, 1))

# # As an example
# x = np.random.rand(2, 1)

# theta_hat = np.matmul(np.linalg.inv(A), b)
# # Calculate the reward point estimate and UCB
# delta = 0.005
# alpha = 1 + np.sqrt(np.log(2 / delta) / 2)
# point_est = np.matmul(np.transpose(theta_hat), x)[(0, 0)]
# ucb = (
#     alpha * np.sqrt(np.matmul(np.matmul(np.transpose(x), np.linalg.inv(A)), x))[(0, 0)]
# )
# p = point_est + ucb

# # Get reward from arm
# a_t = np.argmax(p)
# r_t = reward_bernoulli(a_t)

# # Update algorithm parameters
# A = A + np.matmul(x, np.transpose(x))
# b = b + (r_t * x)

# ### Hybrid model example

# k = 2
# d = 4

# A = np.identity(n=k)
# b = np.zeros((k, 1))

# # Example feature arrays
# z = np.random.rand(k, 1)
# x = np.random.rand(d, 1)

# # Shared parameters
# beta_hat = np.matmul(np.linalg.inv(A), b)

# # Arm-level estimation

# # If a \in A is new
# A_arm = np.identity(n=d)
# B_arm = np.zeros((d, k))
# b_arm = np.zeros((d, 1))

# # Else ...
# theta_hat_arm = np.matmul(np.linalg.inv(A_arm), b_arm - np.matmul(B_arm, beta_hat))
# ucb_arm = (
#     np.matmul(np.matmul(np.transpose(z), np.linalg.inv(A)), z)
#     - 2
#     * np.matmul(
#         np.matmul(
#             np.matmul(
#                 np.matmul(np.transpose(z), np.linalg.inv(A)), np.transpose(B_arm)
#             ),
#             np.linalg.inv(A_arm),
#         ),
#         x,
#     )
#     + np.matmul(np.matmul(np.transpose(x), np.linalg.inv(A_arm)), x)
#     + np.matmul(
#         np.matmul(
#             np.matmul(
#                 np.matmul(
#                     np.matmul(np.matmul(np.transpose(x), np.linalg.inv(A_arm)), B_arm),
#                     np.linalg.inv(A),
#                 ),
#                 np.transpose(B_arm),
#             ),
#             np.linalg.inv(A_arm),
#         ),
#         x,
#     )
# )
# p_arm = (
#     np.matmul(np.transpose(z), beta_hat)
#     + np.matmul(np.transpose(x), theta_hat_arm)
#     + alpha * np.sqrt(ucb_arm)
# )[(0, 0)]

# # Get reward from arm
# a_t = np.argmax(p_arm)
# r_t = reward_bernoulli(a_t)

# # Update universal algorithm parameters
# A = A + np.matmul(np.matmul(np.transpose(B_arm), np.linalg.inv(A_arm)), B_arm)
# b = b + np.matmul(np.matmul(np.transpose(B_arm), np.linalg.inv(A_arm)), b_arm)

# # Update arm-specific parameters
# A_arm = A_arm + np.matmul(x, np.transpose(x))
# B_arm = B_arm + np.matmul(x, np.transpose(z))
# b_arm = b_arm + r_t * x

# # Finish updating universal algorithm parameters
# A = (
#     A
#     + np.matmul(z, np.transpose(z))
#     - np.matmul(np.matmul(np.transpose(B_arm), np.linalg.inv(A_arm)), B_arm)
# )
# b = (
#     b
#     + (r_t * z)
#     - np.matmul(np.matmul(np.transpose(B_arm), np.linalg.inv(A_arm)), b_arm)
# )

import numpy as np
import numpy.lib.recfunctions as npr
import numpy.typing as npt
import pandas as pd
from pathlib import Path
import plotnine as pn
from tqdm import tqdm
from typing import Any, Callable, Iterable, List, Optional

base_dir = Path(__file__).resolve().parent.parent
# base_dir = Path().resolve()


class LinUCB:
    """
    This is a parent class for the two UCB algorithms devised by Li et al.:
    https://arxiv.org/abs/1003.0146.

    This algorithm is designed for optimal assignment in contextual bandits,
    or multi-armed bandits that contain additional information about the arms and/or
    respondents. Contextual bandits encapsulate traditional multi-arm bandits,
    as these can be thought of as contextual bandits with a constant context.
    """

    def __init__(
        self, alpha: float, horizon: int, arms: npt.NDArray[np.record], arms_index: str
    ):
        self.alpha = alpha
        self.horizon = horizon
        self.arms = np.rec.array(
            npr.append_fields(
                arms,
                names="active",
                data=np.ones(arms.shape[0]),
                dtypes=np.bool_,
                usemask=False,
            )
        )
        self.arms_index = arms_index
        self.selected_arms = np.empty(0)
        self.rewards = np.empty(0)
    
    def _activate_arm(self, index_val: str) -> None:
        """Mark an existing bandit arm as active"""
        self.arms["active"][self.arms[self.arms_index] == index_val] = True

    def _active_arms(self) -> npt.NDArray[np.record]:
        """An array of arms that are currently active"""
        return self.arms[self.arms["active"]]

    def _add_arm(self, arm: np.record) -> None:
        """Add a new arm to the bandit"""
        arm = npr.append_fields(
            arm, names="active", data=np.ones(1), dtypes=np.bool_, usemask=False
        )
        self.arms = np.rec.array(np.append(self.arms, arm))

    def _arm_exists(self, index_val: str) -> bool:
        """Check if an arm already exists"""
        return index_val in self.arms[self.arms_index]

    def _deactivate_arm(self, index_val: str) -> None:
        """Mark an existing bandit arm as inactive"""
        self.arms["active"][self.arms[self.arms_index] == index_val] = False

    def _get_arm_features(self, index_val: str) -> np.record:
        """Get an arm features as an array"""
        features = self.arms[self.arms[self.arms_index] == index_val]
        features = npr.drop_fields(
            features,
            [self.arms_index, "active"],
            usemask=False, 
            asrecarray=True
        )
        assert (
            len(features) == 1
        ), f"Exactly one arm was expected but {len(features)} {'were' if len(features) != 1 else 'was'} found"
        return features[0]
    
    def _inactive_arms(self) -> npt.NDArray[np.record]:
        """An array of arms that are currently inactive"""
        return self.arms[~self.arms["active"]]


class LinUCBDisjoint(LinUCB):
    """
    This class implements one of the two primary LinUCB algorithms devised by
    Li et al. LinUCBDisjoint is said to be disjoint because the conditional
    reward function in each arm is linear in its feature vector x_{t,a} with
    an unknown coefficient vector theta_a. That is, at time t:
    E[r_{t, a} | x_{t, a}] = (x_{t,a})^T * theta_a. It is disjoint because the
    unknown coefficient vector theta_a is estimated separately for each arm a.
    Intuitively, this means we are imposing structure on the conditional reward
    function at the arm level. This means we are sharing information across
    all contexts that are assigned to a specific arm, but not sharing contextual
    information across arms.
    """

    def __init__(
        self, alpha: float, horizon: int, arms: npt.NDArray[np.record], arms_index: str
    ):
        super().__init__(alpha, horizon, arms, arms_index)
        params_dtype = [
            (self.arms_index, "O"),
            ("A", "O"),
            ("b", "O"),
            ("context", "O"),
            ("ub", "f8"),
            ("is_new", "bool")
        ]
        params_array = np.array(np.empty(0), dtype=params_dtype)
        for key in self.arms[self.arms_index]:
            new_params = np.array(
                [(
                    key,
                    np.empty((0, 0)),
                    np.empty((0, 0)),
                    np.empty(0),
                    np.nan,
                    True
                )],
                dtype=params_dtype
            )
            params_array = np.append(params_array, new_params)
        self._arm_parameters = params_array
        self._d = None
        self._feature_dtype = None

    def _add_arm(self, arm: np.record) -> None:
        super()._add_arm(arm)
        params_dtype = [
            (self.arms_index, "O"),
            ("A", "O"),
            ("b", "O"),
            ("context", "O"),
            ("ub", "f8"),
            ("is_new", "bool")
        ]
        new_key = arm[self.arms_index]
        new_params = np.array(
            [(
                new_key,
                np.empty((0, 0)),
                np.empty((0, 0)),
                np.empty(0),
                np.nan,
                True
            )],
            dtype=params_dtype
        )
        self._arm_parameters = np.append(self._arm_parameters, new_params)

    def _combine_features(
        self, arm_features: np.record, user_features: np.record
    ) -> np.record:
        features = np.record(
            (arm_features.item() + user_features.item()), dtype=self._feature_dtype
        )
        return features

    def _init_d(self, user_features: np.record) -> None:
        if self._d is None:
            self._d = len(user_features)

    def _init_feature_dtype(self, user_features: np.record) -> None:
        if self._feature_dtype is None:
            arm_dtype = (
                npr
                .drop_fields(self.arms, [self.arms_index, "active"], usemask=False)
                .dtype
                .descr
            )
            user_dtype = user_features.dtype.descr
            self._feature_dtype = arm_dtype + user_dtype

    def _init_arm_parameters(self) -> None:
        new_arms = self._arm_parameters[self._arm_parameters["is_new"]]
        for arm in new_arms:
            arm["A"] = np.identity(n=self._d, dtype=np.float64)
            arm["b"] = np.zeros(shape=(self._d, 1), dtype=np.float64)
            arm["is_new"] = False
        self._arm_parameters[self._arm_parameters["is_new"]] = new_arms

    def pull(self, user_features: np.record, reward: Callable[[Any], float], logging_index_val: Any | None = None) -> Any:
        """Run one iteration of the algorithm"""

        # Estimate parameters and upper confidence bounds for each active arm
        for arm_index in self._active_arms()[self.arms_index]:

            # Retrieve full context vector
            self._init_feature_dtype(user_features)
            context = self._combine_features(
                arm_features=self._get_arm_features(index_val=arm_index),
                user_features=user_features,
            )

            # Ensure dimension and arm design matrices are initiated
            self._init_d(context)
            self._init_arm_parameters()

            # Pre-calculate inverses and such
            parameters = self._arm_parameters[self._arm_parameters[self.arms_index] == arm_index]
            assert len(parameters) == 1, f"More than one set of parameters found for Arm {arm_index}"
            for parameter in parameters:
                A = parameter["A"]
                A_inv = np.linalg.inv(A)
                b = parameter["b"]
                x_t = np.array(context.item()).reshape(1, -1)
                x = np.transpose(x_t)

            # Estimate parameters and upper bound
            theta_hat = np.matmul(A_inv, b)
            ub = (
                np.matmul(np.transpose(theta_hat), x)
                + self.alpha * np.sqrt(np.matmul(np.matmul(x_t, A_inv), x))
            )[(0, 0)]

            # Update parameter values
            for parameter in parameters:
                parameter["context"] = x
                parameter["ub"] = ub
            self._arm_parameters[self._arm_parameters[self.arms_index] == arm_index] = parameters
        
        # Select the arm with the highest upper bound (ties broken randomly)
        ubs = self._arm_parameters["ub"]
        optimal_arm: int = np.random.choice(np.flatnonzero(ubs == ubs.max()))
        optimal_arm_index = self._arm_parameters[self.arms_index][optimal_arm]

        # Return without updating parameters (effectively do not append records to history)
        if logging_index_val is not None and optimal_arm_index != logging_index_val:
            return optimal_arm_index
        
        # Observe reward
        r = reward(optimal_arm_index)

        # Update parameters for selected arm
        parameters = self._arm_parameters[self._arm_parameters[self.arms_index] == optimal_arm_index]
        assert len(parameters) == 1, f"More than one set of parameters found for Arm {optimal_arm_index}"
        for parameter in parameters:
            parameter["A"] = (
                parameter["A"]
                + np.matmul(parameter["context"], np.transpose(parameter["context"]))
            )
            parameter["b"] = parameter["b"] + r*parameter["context"]
        self._arm_parameters[self._arm_parameters[self.arms_index] == optimal_arm_index] = parameters

        # Update reward and selected arms
        self.selected_arms = np.append(self.selected_arms, optimal_arm_index)
        self.rewards = np.append(self.rewards, r)

        return optimal_arm_index


class LinUCBHybrid(LinUCB):
    """
    TODO
    """

    def __init__(
        self, alpha: float, horizon: int, arms: npt.NDArray[np.record], arms_index: str
    ):
        super().__init__(alpha, horizon, arms, arms_index)
        params_dtype = [
            (self.arms_index, "O"),
            ("A", "O"),
            ("b", "O"),
            ("context", "O"),
            ("ub", "f8"),
            ("is_new", "bool")
        ]
        params_array = np.array(np.empty(0), dtype=params_dtype)
        for key in self.arms[self.arms_index]:
            new_params = np.array(
                [(
                    key,
                    np.empty((0, 0)),
                    np.empty((0, 0)),
                    np.empty(0),
                    np.nan,
                    True
                )],
                dtype=params_dtype
            )
            params_array = np.append(params_array, new_params)
        self._arm_parameters = params_array
        self._d = None
        self._feature_dtype = None
        self._A = np.empty((0, 0))
        self._b = np.empty((0, 0))

    def _add_arm(self, arm: np.record) -> None:
        super()._add_arm(arm)
        params_dtype = [
            (self.arms_index, "O"),
            ("A", "O"),
            ("b", "O"),
            ("context", "O"),
            ("ub", "f8"),
            ("is_new", "bool")
        ]
        new_key = arm[self.arms_index]
        new_params = np.array(
            [(
                new_key,
                np.empty((0, 0)),
                np.empty((0, 0)),
                np.empty(0),
                np.nan,
                True
            )],
            dtype=params_dtype
        )
        self._arm_parameters = np.append(self._arm_parameters, new_params)

    def _combine_features(
        self, arm_features: np.record, user_features: np.record
    ) -> np.record:
        features = np.record(
            (arm_features.item() + user_features.item()), dtype=self._feature_dtype
        )
        return features

    def _init_d(self, user_features: np.record) -> None:
        if self._d is None:
            self._d = len(user_features)
    
    def _init_k(self, shared_features: np.record) -> None:
        if self._k is None:
            self._k = len(shared_features)

    def _init_feature_dtype(self, user_features: np.record) -> None:
        if self._feature_dtype is None:
            arm_dtype = (
                npr
                .drop_fields(self.arms, [self.arms_index, "active"], usemask=False)
                .dtype
                .descr
            )
            user_dtype = user_features.dtype.descr
            self._feature_dtype = arm_dtype + user_dtype

    def _init_arm_parameters(self) -> None:
        new_arms = self._arm_parameters[self._arm_parameters["is_new"]]
        for arm in new_arms:
            arm["A"] = np.identity(n=self._d, dtype=np.float64)
            arm["b"] = np.zeros(shape=(self._d, 1), dtype=np.float64)
            arm["is_new"] = False
        self._arm_parameters[self._arm_parameters["is_new"]] = new_arms

    def pull(
        self,
        user_features: np.record,
        shared_features: List[str],
        reward: Callable[[Any], float],
        logging_index_val: Any | None = None
    ) -> Any:
        """Run one iteration of the algorithm"""

        # Estimate parameters and upper confidence bounds for each active arm
        for arm_index in self._active_arms()[self.arms_index]:

            # Retrieve full context vector
            self._init_feature_dtype(user_features)
            context = self._combine_features(
                arm_features=self._get_arm_features(index_val=arm_index),
                user_features=user_features,
            )

            # Split out disjoint and shared features
            ## TODO keep working on this!
            shared_features = context[shared_features]
            disjoint_features = npr.drop_fields(context, shared_features, usemask=False, asrecarray=True)

            # Ensure dimension and arm design matrices are initiated
            self._init_d(context)
            self._init_arm_parameters()

            # Pre-calculate inverses and such
            parameters = self._arm_parameters[self._arm_parameters[self.arms_index] == arm_index]
            assert len(parameters) == 1, f"More than one set of parameters found for Arm {arm_index}"
            for parameter in parameters:
                A = parameter["A"]
                A_inv = np.linalg.inv(A)
                b = parameter["b"]
                x_t = np.array(context.item()).reshape(1, -1)
                x = np.transpose(x_t)

            # Estimate parameters and upper bound
            theta_hat = np.matmul(A_inv, b)
            ub = (
                np.matmul(np.transpose(theta_hat), x)
                + self.alpha * np.sqrt(np.matmul(np.matmul(x_t, A_inv), x))
            )[(0, 0)]

            # Update parameter values
            for parameter in parameters:
                parameter["context"] = x
                parameter["ub"] = ub
            self._arm_parameters[self._arm_parameters[self.arms_index] == arm_index] = parameters
        
        # Select the arm with the highest upper bound (ties broken randomly)
        ubs = self._arm_parameters["ub"]
        optimal_arm: int = np.random.choice(np.flatnonzero(ubs == ubs.max()))
        optimal_arm_index = self._arm_parameters[self.arms_index][optimal_arm]

        # Return without updating parameters (effectively do not append records to history)
        if logging_index_val is not None and optimal_arm_index != logging_index_val:
            return optimal_arm_index
        
        # Observe reward
        r = reward(optimal_arm_index)

        # Update parameters for selected arm
        parameters = self._arm_parameters[self._arm_parameters[self.arms_index] == optimal_arm_index]
        assert len(parameters) == 1, f"More than one set of parameters found for Arm {optimal_arm_index}"
        for parameter in parameters:
            parameter["A"] = (
                parameter["A"]
                + np.matmul(parameter["context"], np.transpose(parameter["context"]))
            )
            parameter["b"] = parameter["b"] + r*parameter["context"]
        self._arm_parameters[self._arm_parameters[self.arms_index] == optimal_arm_index] = parameters

        # Update reward and selected arms
        self.selected_arms = np.append(self.selected_arms, optimal_arm_index)
        self.rewards = np.append(self.rewards, r)

        return optimal_arm_index
