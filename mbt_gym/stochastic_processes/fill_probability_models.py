import abc
from typing import Optional, Tuple

import numpy as np
import math
import matplotlib.pyplot as plt
from mbt_gym.stochastic_processes.StochasticProcessModel import StochasticProcessModel
from mbt_gym.gym.index_names import *


class FillProbabilityModel(StochasticProcessModel):
    def __init__(
        self,
        min_value: np.ndarray,
        max_value: np.ndarray,
        step_size: float,
        terminal_time: float,
        initial_state: np.ndarray,
        num_trajectories: int = 1,
        seed: int = None,
    ):
        super().__init__(min_value, max_value, step_size, terminal_time, initial_state, num_trajectories, seed)

    @abc.abstractmethod
    def _get_fill_probabilities(self, depths: np.ndarray) -> np.ndarray:
        """Note that _get_fill_probabilities can return a 'probability' greater than one. However, this is not an issue
        for it is only use is in `get_hypothetical_fills` below."""
        pass

    def get_fills(self, depths: np.ndarray) -> np.ndarray:
        assert depths.shape == (self.num_trajectories, 2), (
            "Depths must be a numpy array of shape "
            + f"({self.num_trajectories},2). Instead it is a numpy array of shape {depths.shape}."
        )
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        #print(f'Unif= {unif}')
        return unif < self._get_fill_probabilities(depths)

    @property
    @abc.abstractmethod
    def max_depth(self) -> float:
        pass


class ExponentialFillFunction(FillProbabilityModel):
    def __init__(
        self, fill_exponent: float = 1.5, step_size: float = 0.1, num_trajectories: int = 1, seed: Optional[int] = None
    ):
        self.fill_exponent = fill_exponent
        super().__init__(
            min_value=np.array([[]]),
            max_value=np.array([[]]),
            step_size=step_size,
            terminal_time=0.0,
            initial_state=np.array([[]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def _get_fill_probabilities(self, depths: np.ndarray) -> np.ndarray:
        return np.exp(-self.fill_exponent * depths)       

    @property
    def max_depth(self) -> float:
        return -np.log(0.01) / self.fill_exponent

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        #print(f'Arrivals = {arrivals}')
        pass


class StochasticExponentialFillFunction(ExponentialFillFunction):
    def __init__(
        self,
        baseline_kappa: np.ndarray = np.array([[1.5, 1.5]]),
        step_size: float = 0.1,
        opp_side_jump: float = 1,  # νκ 
        same_side_jump: float = 2.5,   # ηκ
        mean_reversion_speed_kappa: float = 50,
        num_trajectories: int = 1,
        p: float = 0.7,  # Influential order threshold
        seed: Optional[int] = None
    ):
        self.baseline_kappa = baseline_kappa
        self.opp_side_jump = opp_side_jump
        self.same_side_jump = same_side_jump
        self.mean_reversion_speed_kappa = mean_reversion_speed_kappa
        self.current_kappa = np.copy(baseline_kappa)
        self.p = p  # Influential order threshold
        self.num_trajectories = num_trajectories
        
        if seed is not None:
            np.random.seed(seed)

        super().__init__(step_size=step_size, num_trajectories=num_trajectories, seed=seed)

    def is_influential(self) -> np.ndarray:
        """Determine if an order is influential for each trajectory and for both sides"""
        return np.random.uniform(size=(self.num_trajectories, 2)) < self.p

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> None:

        # Check if the order is influential for each trajectory and for both sides
        influential_orders = self.is_influential()

        # Modify arrivals based on whether each trajectory's order is influential
        arrivals *= influential_orders

        # Separate the arrivals into sells and buys
        sell_arrivals, buy_arrivals = arrivals[:, BID_INDEX], arrivals[:, ASK_INDEX]

        # Compute the jumps for both sides
        jump_sell_side = np.array([[self.same_side_jump, self.opp_side_jump]]) * sell_arrivals.reshape(-1, 1)
        jump_buy_side = np.array([[self.opp_side_jump, self.same_side_jump]]) * buy_arrivals.reshape(-1, 1)

        # Combine the jumps
        total_jump = jump_sell_side + jump_buy_side

        # Mean reversion term 
        mean_reversion_term = self.mean_reversion_speed_kappa * (np.ones((self.num_trajectories, 2)) * self.baseline_kappa - self.current_kappa) * self.step_size

        # Update current_kappa with both the mean reversion term and the jump term
        self.current_kappa = self.current_kappa + mean_reversion_term + total_jump

        # Ensure kappa values stay non-negative
        self.current_kappa = np.maximum(self.current_kappa, 0)

    def _get_fill_probabilities(self, depths: np.ndarray) -> np.ndarray:
        return np.exp(-self.current_kappa * depths)


class TriangularFillFunction(FillProbabilityModel):
    def __init__(
        self, max_fill_depth: float = 1.0, step_size: float = 0.1, num_trajectories: int = 1, seed: Optional[int] = None
    ):
        self.max_fill_depth = max_fill_depth
        super().__init__(
            min_value=np.array([[]]),
            max_value=np.array([[]]),
            step_size=step_size,
            terminal_time=0.0,
            initial_state=np.array([[]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def _get_fill_probabilities(self, depths: np.ndarray) -> np.ndarray:
        return np.max(1 - np.max(depths, 0) / self.max_fill_depth, 0)

    @property
    def max_depth(self) -> float:
        return 1.5 * self.max_fill_depth

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        pass


class PowerFillFunction(FillProbabilityModel):
    def __init__(
        self,
        fill_exponent: float = 1.5,
        fill_multiplier: float = 1.5,
        step_size: float = 0.1,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.fill_exponent = fill_exponent
        self.fill_multiplier = fill_multiplier
        super().__init__(
            min_value=np.array([[]]),
            max_value=np.array([[]]),
            step_size=step_size,
            terminal_time=0.0,
            initial_state=np.array([[]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def _get_fill_probabilities(self, depths: np.ndarray) -> np.ndarray:
        return (1 + (self.fill_multiplier * np.max(depths, 0)) ** self.fill_exponent) ** -1

    @property
    def max_depth(self) -> float:
        return 0.01 ** (-1 / self.fill_exponent) - 1

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        pass


class ExogenousMmFillProbabilityModel(FillProbabilityModel):
    def __init__(
        self,
        exogenous_best_depth_processes: Tuple[StochasticProcessModel],
        fill_exponent: float = 1.5,
        base_fill_probability: float = 1.0,
        step_size: float = 0.1,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        assert len(exogenous_best_depth_processes) == 2, "exogenous_best_depth_processes must be length 2 (bid and ask)"
        assert all(
            len(process.initial_state) > 0 for process in exogenous_best_depth_processes
        ), "Exogenous best depth processes must have a state of at least size 1."
        self.exogenous_best_depth_processes = exogenous_best_depth_processes
        self.fill_exponent = fill_exponent
        self.base_fill_probability = base_fill_probability
        super().__init__(
            min_value=np.concatenate([process.min_value for process in self.exogenous_best_depth_processes], axis=1),
            max_value=np.concatenate([process.max_value for process in self.exogenous_best_depth_processes], axis=1),
            step_size=step_size,
            terminal_time=0.0,
            initial_state=np.concatenate(
                (
                    self.exogenous_best_depth_processes[0].initial_state,
                    self.exogenous_best_depth_processes[1].initial_state,
                ),
                axis=1,
            ),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def _get_fill_probabilities(self, depths: np.ndarray) -> np.ndarray:
        return (depths > self.current_state) * self.base_fill_probability * np.exp(
            -self.fill_exponent * (depths - self.current_state)
        ) + (depths <= self.current_state)

    @property
    def max_depth(self) -> float:
        return -np.log(0.01) / self.fill_exponent + np.max(self.exogenous_best_depth_processes[0].max_value)

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        for process in self.exogenous_best_depth_processes:
            process.update(arrivals, fills, actions)
