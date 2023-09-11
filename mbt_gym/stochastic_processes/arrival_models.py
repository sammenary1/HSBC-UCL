import abc
from typing import Optional

import numpy as np

from mbt_gym.stochastic_processes.StochasticProcessModel import StochasticProcessModel


class ArrivalModel(StochasticProcessModel):
    """ArrivalModel models the arrival of orders to the order book. The first entry of arrivals represents an arrival
    of an exogenous SELL order (arriving on the buy side of the book) and the second entry represents an arrival of an
    exogenous BUY order (arriving on the sell side of the book).
    """

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
    def get_arrivals(self) -> np.ndarray:
        pass


class PoissonArrivalModel(ArrivalModel):
    def __init__(
        self,
        intensity: np.ndarray = np.array([140.0, 140.0]),
        step_size: float = 0.001,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.intensity = np.array(intensity)
        super().__init__(
            min_value=np.array([[]]),
            max_value=np.array([[]]),
            step_size=step_size,
            terminal_time=0.0,
            initial_state=np.array([[]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        pass

    def get_arrivals(self) -> np.ndarray:
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        return unif < self.intensity * self.step_size


class HawkesArrivalModel(ArrivalModel):
    def __init__(
        self,
        baseline_arrival_rate: np.ndarray = np.array([[10.0, 10.0]]),
        step_size: float = 0.01,
        jump_size: float = 40.0,
        mean_reversion_speed: float = 60.0,
        terminal_time: float = 1,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.baseline_arrival_rate = baseline_arrival_rate
        self.jump_size = jump_size  # see https://arxiv.org/pdf/1507.02822.pdf, equation (4).
        self.mean_reversion_speed = mean_reversion_speed
        super().__init__(
            min_value=np.array([[0, 0]]),
            max_value=np.array([[1, 1]]) * self._get_max_arrival_rate(),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=baseline_arrival_rate,
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
   
        self.current_state = (
            self.current_state
            + self.mean_reversion_speed
            * (np.ones((self.num_trajectories, 2)) * self.baseline_arrival_rate - self.current_state)
            * self.step_size
            * np.ones((self.num_trajectories, 2))
            + self.jump_size * arrivals
        )
        print(f'arrivals/current_state = {self.current_state}')
        return self.current_state

    def get_arrivals(self) -> np.ndarray:
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        return unif < self.current_state * self.step_size

    def _get_max_arrival_rate(self):
        return self.baseline_arrival_rate * 10

    # TODO: Improve this with 4*std
    # See: https://math.stackexchange.com/questions/4047342/expectation-of-hawkes-process-with-exponential-kernel
    
# class BuyLowSellHigh(ArrivalModel):
#     def __init__(self, arrival_rate_mean: float = 1.5, mean_reversion_speed: float = 1.5,
#                  buy_jump_activity: float = 1, sell_jump_activity: float = 1.5,
#                  terminal_time: float = 1.0, step_size: float = 0.1, num_trajectories: int = 1,
#                  seed: Optional[int] = None):
#         self.arrival_rate_mean = arrival_rate_mean
#         self.mean_reversion_speed = mean_reversion_speed
#         self.buy_jump_activity = buy_jump_activity #n
#         self.buy_indicator = np.zeros(num_trajectories, dtype=int)  # Initialize buy_indicator for each trajectory
#         self.sell_jump_activity = sell_jump_activity #v
#         self.arrival_rate: np.ndarray = np.array([[10.0, 10.0]]),
#         self.terminal_time = terminal_time
#         super().__init__(
#             min_value=np.array([0]),
#             max_value=np.array([np.inf]),
#             step_size=step_size,
#             terminal_time=terminal_time,
#             initial_state=np.array([arrival_rate_mean]),
#             num_trajectories=num_trajectories,
#             seed=seed,
#         )
        
#     def get_arrivals(self) -> np.ndarray:
#         unif = self.rng.uniform(size=(self.num_trajectories, 2))
#         return unif < self.current_state * self.step_size

#     def _get_max_arrival_rate(self):
#         return self.baseline_arrival_rate * 10 #?
    
#     def is_influential(self):
#         unif = self.rng.uniform(size=(self.num_trajectories, 2))
#         binary = (unif < 0.7).astype(int)
#         print(f'is_influential = \n{binary}')
#         return binary
    
#     def time_until_next_trade(self):
        
    
#     def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
#         print(f'Arrivals = \n{arrivals}')
#         for i in range(self.num_trajectories):
#             if arrivals[i][0] and arrivals[i][1]:  # If both arrivals[i][0] and arrivals[i][1] are True
#                 influential_indicator_0 = self.is_influential()[i][0]
#                 influential_indicator_1 = self.is_influential()[i][1]
#                 jump_activity_0 = (
#                     0.5 * (1 - self.buy_indicator[i]) * self.sell_jump_activity
#                     + 0.5 * (1 + self.buy_indicator[i]) * self.buy_jump_activity
#                 )
#                 jump_activity_1 = (
#                     0.5 * (1 + self.buy_indicator[i]) * self.sell_jump_activity
#                     + 0.5 * (1 - self.buy_indicator[i]) * self.buy_jump_activity
#                 )

#                 self.arrival_rate[i][0] = (
#                     self.arrival_rate_mean
#                     + (self.arrival_rate[i][0] - self.arrival_rate_mean) * math.exp(-self.mean_reversion_speed * self.terminal_time)
#                     + jump_activity_0 * influential_indicator_0
#                     + jump_activity_1 * influential_indicator_1
#                 )

#                 self.arrival_rate[i][1] = (
#                     self.arrival_rate_mean
#                     + (self.arrival_rate[i][1] - self.arrival_rate_mean) * math.exp(-self.mean_reversion_speed * self.terminal_time)
#                     + jump_activity_1 * influential_indicator_1
#                     + jump_activity_0 * influential_indicator_0
#                 )
#             elif arrivals[i][0] == True:
#                 influential_indicator = self.is_influential()[i][0]
#                 self.buy_indicator[i] = -1  # set buy_indicator to -1 (Sell MO)
#                 jump_activity = (
#                     0.5 * (1 - self.buy_indicator[i]) * self.sell_jump_activity
#                     + 0.5 * (1 + self.buy_indicator[i]) * self.buy_jump_activity
#                 )
#                 print(f'jump_activity {jump_activity}')
#                 self.arrival_rate[i][0] = (
#                     self.arrival_rate_mean
#                     + (self.arrival_rate[i][0] - self.arrival_rate_mean) * math.exp(-self.mean_reversion_speed * self.terminal_time)
#                     + jump_activity * influential_indicator
#                 )
#                 print(f'arrival_rate = {self.arrival_rate[i][0]}')
#                 jump_activity = (
#                     0.5 * (1 + self.buy_indicator[i]) * self.sell_jump_activity
#                     + 0.5 * (1 - self.buy_indicator[i]) * self.buy_jump_activity
#                 )
#                 print(f'jump_activity {jump_activity}')
#                 self.arrival_rate[i][1] = (
#                     self.arrival_rate_mean
#                     + (self.arrival_rate[i][1] - self.arrival_rate_mean) * math.exp(-self.mean_reversion_speed * self.terminal_time)
#                     + jump_activity * influential_indicator
#                 )
#                 print(f'arrival rate = {self.arrival_rate[i][1]}')
                
#             elif arrivals[i][1] == True:
#                 influential_indicator = self.is_influential()[i][1]
#                 self.buy_indicator[i] = 1  # set buy_indicator to 1 (Buy MO)
#                 jump_activity = (
#                     0.5 * (1 + self.buy_indicator[i]) * self.sell_jump_activity
#                     + 0.5 * (1 - self.buy_indicator[i]) * self.buy_jump_activity
#                 )
#                 self.arrival_rate[i][1] = (
#                     self.arrival_rate_mean
#                     + (self.arrival_rate[i][1] - self.arrival_rate_mean) * math.exp(-self.mean_reversion_speed * self.terminal_time)
#                     + jump_activity * influential_indicator
#                 )

#                 jump_activity = (
#                     0.5 * (1 - self.buy_indicator[i]) * self.sell_jump_activity
#                     + 0.5 * (1 + self.buy_indicator[i]) * self.buy_jump_activity
#                 )
#                 self.arrival_rate[i][0] = (
#                     self.arrival_rate_mean
#                     + (self.arrival_rate[i][0] - self.arrival_rate_mean) * math.exp(-self.mean_reversion_speed * self.terminal_time)
#                     + jump_activity * influential_indicator
#                 )
#             else:
#                 self.arrival_rate[i][0] = (
#                     self.arrival_rate_mean
#                     + (self.arrival_rate[i][0] - self.arrival_rate_mean) * math.exp(-self.mean_reversion_speed * self.terminal_time)
#                 )
                
#                 self.arrival_rate[i][1] = (
#                     self.arrival_rate_mean
#                     + (self.arrival_rate[i][1] - self.arrival_rate_mean) * math.exp(-self.mean_reversion_speed * self.terminal_time)
#                 )

#             # After updating kappa based on various conditions, clip the values to enforce min_value and max_value
#             self.arrival_rate[:, 0] = np.clip(self.arrival_rate[:, 0], self.min_value[:, 0], self.max_value[:, 0])
#             self.arrival_rate[:, 1] = np.clip(self.arrival_rate[:, 1], self.min_value[:, 1], self.max_value[:, 1])

#         print('NEW TIMESTEP')
