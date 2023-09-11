from yahoofinancials import YahooFinancials 
from datetime import date, timedelta, datetime
import numpy as np

class MarketData:
    def __init__(self, currency_pair):
        self.currency_pair = currency_pair
        self.yahoo_financials = YahooFinancials(currency_pair)

    def get_initial_price(self, date):
        try:
            historical_data = self.yahoo_financials.get_historical_price_data(date, date, 'daily')
            opening_price = historical_data[self.currency_pair]['prices'][0]['open']
            return opening_price
        except Exception as e:
            print(f"Error getting initial price for date, please select a valid date and try again. {date}: {e}")
            return None

    
    def get_volatility(self, date):
        try:
            end_date = datetime.strptime(date, "%Y-%m-%d")
            start_date = end_date - timedelta(days=21)
            historical_data = self.yahoo_financials.get_historical_price_data(start_date.strftime("%Y-%m-%d"),
                                                                             end_date.strftime("%Y-%m-%d"), 'daily')
            prices = [data['open'] for data in historical_data[self.currency_pair]['prices']]
            average_price = np.mean(prices)
            differences = [(price - average_price) for price in prices]
            squared_differences = [difference ** 2 for difference in differences]
            variance = np.sum(squared_differences) / len(prices)
            volatility = np.sqrt(variance)
            return volatility
        except Exception as e:
            print(f"Error calculating volatility: {e}")
            return None

##Add these lines to use in 'learning to make a market...' notebook

#from mbt_gym.gym.market_data import MarketData

#def get_cj_env(currency_pair: str, date: str, num_trajectories: int = 1):

    #market_data = MarketData(currency_pair)
    #initial_price = market_data.get_initial_price(date)
    #sigma = market_data.get_volatility(date)
    
    #same as before...

#num_trajectories = 1000
#currency_pair = 'EURUSD=X'
#date = '2023-06-30'
#env = ReduceStateSizeWrapper(get_cj_env(currency_pair, date, num_trajectories=num_trajectories))
#sb_env = StableBaselinesTradingEnvironment(trading_env=env)
