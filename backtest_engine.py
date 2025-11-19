import pandas as pd
import numpy as np

class framework:
    def __init__(self, initial_cash=0):
        self.df = pd.DataFrame()  
        self.df['Total_equity'] = 0
        self.df['Cash'] = 0
        self.strategies = []
        self.results = {}
        self.cash = initial_cash
        self.equity = self.cash
        self.analyse_tool = self.AnalyseTool(self)
        self.ticker_lst = []
    
    def initial_cash(self, initial_cash):
        self.cash  = initial_cash
        self.equity = self.cash

    def add_data(self, dataname, data):
        self.ticker = dataname
        self.df = pd.concat([self.df, data], axis=1)
        self.df = self.df.rename(columns={'Close': dataname})
        self.df['Cash'] = self.df['Cash'].fillna(self.cash)
        self.df['Total_equity'] = self.df['Total_equity'].fillna(self.cash)
        self.df[dataname + '_holding_position'] = 0.0
        self.df[dataname + '_average_cost'] = 0.0
        self.df[dataname + '_commission'] = 0.0
        self.df[dataname + '_open_signal'] = 0.0
        self.df[dataname + '_close_signal'] = 0.0
        self.df[dataname + '_holding_market_value'] = 0.0
        self.df[dataname + '_holding_PnL'] = 0.0
        self.df[dataname + '_holding_period'] = 0.0
        self.df[dataname + '_initial_margin'] = 0.0
        self.df[dataname + '_long_trade'] = 0.0
        self.df[dataname + '_long_win'] = 0.0
        self.df[dataname + '_long_win_rate'] = 0.0
        self.df[dataname + '_short_trade'] = 0.0
        self.df[dataname + '_short_win'] = 0.0
        self.df[dataname + '_short_win_rate'] = 0.0
        self.results[dataname] = {'long': [], 'short': []}


    def add_strategy(self, strategy_class, **kwargs):
        strategy_instance = strategy_class(self.df, **kwargs) 
        if hasattr(strategy_instance, 'add_reference_data'):
            print("Adding reference data.")
            strategy_instance.add_reference_data(self, self.ticker)
        else:
            print("add_reference_data method not found.")
        self.strategies.append(strategy_instance)


    def open_position(self, index, ticker, open_price, high, low, buy_qty, direction, strategy_type=None, multiplier = 1):
        last_index = self.df.index[self.df.index.get_loc(index) - 1]
        last_avg_cost = self.df.loc[last_index, f'{ticker}_average_cost']

        if pd.isna(last_avg_cost) or self.df.loc[last_index, f'{ticker}_holding_position'] == 0:
            new_avg_cost = open_price
        else:
            new_avg_cost = (
                last_avg_cost * abs(self.df.loc[last_index, f'{ticker}_holding_position']) + open_price * buy_qty
            ) / (abs(self.df.loc[last_index, f'{ticker}_holding_position']) + buy_qty)
        
        initial_margin = (open_price * buy_qty)/multiplier
        commission = abs(buy_qty)*open_price*0.0005

        if direction == 'long':
            self.df.loc[index, f'{ticker}_holding_position'] = self.df.loc[index, f'{ticker}_holding_position'] + buy_qty
            self.df.loc[index, f'{ticker}_open_signal'] = 1
            self.df.loc[index, 'Cash'] -= initial_margin
            self.df.loc[index, f'{ticker}_initial_margin'] += initial_margin
            self.df.loc[index, f'{ticker}_best_price'] = open_price 
            self.df.loc[index, f'{ticker}_highest_high'] = high 

        elif direction == 'short' :
            self.df.loc[index, f'{ticker}_holding_position'] = self.df.loc[last_index, f'{ticker}_holding_position'] - buy_qty
            self.df.loc[index, f'{ticker}_open_signal'] = -1
            self.df.loc[index, 'Cash'] -= initial_margin
            self.df.loc[index, f'{ticker}_initial_margin'] += initial_margin
            self.df.loc[index, f'{ticker}_best_price'] = open_price
            self.df.loc[index, f'{ticker}_lowest_low'] = low

        self.df.loc[index, f'{ticker}_average_cost'] = new_avg_cost
        self.df.loc[index, f'{ticker}_commission'] += commission
        self.df.loc[index, 'Cash'] -= commission

        if strategy_type is not None:
            self.df.loc[index, 'trading_strategy'] = strategy_type
        self.df.loc[index, f'{ticker}_holding_period'] = 1

            

    def close_position(self, index, ticker, price, buy_qty, action_signal, strategy_type=None):
        last_index = self.df.index[self.df.index.get_loc(index) - 1]
        holding_position = self.df.loc[last_index, f'{ticker}_holding_position']
        realised_pnl = 0
        initial_margin_pre_unit = self.df.loc[index, f'{ticker}_initial_margin']/abs(holding_position)
        commission = abs(buy_qty)*price*0.0005
        if holding_position == 0:
            self.df.loc[index, f'{ticker}_average_cost'] = 0
        if buy_qty > holding_position:
            self.df.loc[index, f'{ticker}_holding_position'] = 0.0
        else:
            self.df.loc[index, f'{ticker}_holding_position'] = holding_position - buy_qty

        net_pnl = 0  # Calculate net PnL
        if action_signal == 1:
            buy_qty = -buy_qty
            realised_pnl = (self.df.loc[index, f'{ticker}_average_cost'] - price)
            cash_delta = (realised_pnl + initial_margin_pre_unit) * abs(buy_qty)
            self.df.loc[index, 'Cash'] += cash_delta
            self.df.loc[index, f'{ticker}_initial_margin'] -= initial_margin_pre_unit* abs(buy_qty)
            self.df.loc[index, f'{ticker}_short_trade'] += 1
            net_pnl = realised_pnl * abs(buy_qty) - (commission * 2)
            if net_pnl > 0:
                self.df.loc[index, f'{ticker}_short_win'] += 1
            self.results[ticker]['short'].append(net_pnl)

        elif action_signal == -1:
            realised_pnl = (price - self.df.loc[index, f'{ticker}_average_cost'])
            cash_delta = (realised_pnl + initial_margin_pre_unit) * abs(buy_qty)
            self.df.loc[index, 'Cash'] += cash_delta
            self.df.loc[index, f'{ticker}_initial_margin'] -= initial_margin_pre_unit* abs(buy_qty)
            self.df.loc[index, f'{ticker}_long_trade'] += 1
            net_pnl = realised_pnl * abs(buy_qty) - (commission * 2)
            if net_pnl > 0:
                self.df.loc[index, f'{ticker}_long_win'] += 1        
            self.df.loc[index, f'{ticker}_commission'] += commission
            self.df.loc[index, 'Cash'] -= commission
            self.results[ticker]['long'].append(net_pnl)


        # win rate calculation 
        long_trades = self.df.loc[index, f'{ticker}_long_trade']
        short_trades = self.df.loc[index, f'{ticker}_short_trade']
        long_win = self.df.loc[index, f'{ticker}_long_win']
        short_win = self.df.loc[index, f'{ticker}_short_win']
        
        if long_trades > 0:
            self.df.loc[index, f'{ticker}_long_win_rate'] = long_win / long_trades
            self.df.loc[index, f'{ticker}_close_signal'] = action_signal
        else:
            self.df.loc[index, f'{ticker}_long_win_rate'] = 0.0

        if short_trades > 0:
            self.df.loc[index, f'{ticker}_short_win_rate'] = short_win / short_trades
            self.df.loc[index, f'{ticker}_close_signal'] = action_signal
        else:
            self.df.loc[index, f'{ticker}_short_win_rate'] = 0.0
        self.df.loc[index, 'trading_strategy'] = strategy_type
        self.df.loc[index, f'{ticker}_holding_period'] = 0
        self.df.loc[index, f'{ticker}_best_price'] = 0.0
        self.df.loc[index, f'{ticker}_highest_high'] = 0.0
        self.df.loc[index, f'{ticker}_lowest_low'] = 0.0






    def run(self):
        for strategy in self.strategies:
            for col in self.df.columns:
                if col[-17:] == '_holding_position':
                    ticker = col.split('_')[0]
                    self.ticker_lst.append(ticker)
                    for i in self.df.index:
                        strategy.next(ticker, self, i)

    def update_position_info(self, index, ticker):
        holding_position = self.df.loc[index, ticker + '_holding_position']
        last_index = self.df.index[self.df.index.get_loc(index) - 1]
        last_holding_position = self.df.loc[last_index, ticker + '_holding_position']
        last_holding_period = self.df.loc[last_index, ticker + '_holding_period']
        price = self.df.loc[index, ticker]
        average_cost = self.df.loc[index, ticker + '_average_cost']

        self.df.loc[index, ticker + '_holding_market_value'] = holding_position * price

        # long
        if holding_position > 0:
            holding_pnl = (price - average_cost) * holding_position 
            if holding_position == last_holding_position:
                self.df.loc[index, ticker+'_holding_period'] = last_holding_period + 1
        # short
        elif holding_position < 0:
            holding_pnl = (average_cost - price) * abs(holding_position) 
            if holding_position == last_holding_position:
                self.df.loc[index, ticker+'_holding_period'] = last_holding_period + 1
        else:
            holding_pnl = 0.0

        self.df.loc[index, ticker + '_holding_PnL'] = holding_pnl
    
    def calculate_return(self):
        self.df['Daily_return'] = self.df['Total_equity'].pct_change().fillna(0)
        self.returns = self.df['Daily_return']

    def reset(self, initial_cash=0):
        self.df = pd.DataFrame() 
        self.strategies = []  
        self.results = {}  
        self.cash = 0  
        self.equity = self.cash 

    def delete_data(self, dataname):
         del_col = [col for col in self.df.columns if dataname in col]
         self.df.drop(columns=del_col, inplace=True)

    class AnalyseTool:
        def __init__(self, backtest):
            self.backtest = backtest

        def sharpe_ratio(self, risk_free_rate=0.043, periods=252):
            risk_free_rate = risk_free_rate 
            annual_return = self.backtest.returns.mean() * periods
            annual_volatility = self.backtest.returns.std() * np.sqrt(periods)
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
            return sharpe_ratio

        def maximum_drawdown(self):
            equity_curve = self.backtest.df['Total_equity']
            drawdown = (equity_curve / equity_curve.cummax()) - 1
            max_drawdown = drawdown.min()
            return max_drawdown

        def correlation_with_ticker(self, ticker):
                    if ticker not in self.backtest.df.columns:
                        raise ValueError(f"Ticker '{ticker}' not found in data.")
                    ticker_returns = self.backtest.df[ticker].pct_change().fillna(0)
                    total_returns = self.backtest.returns
                    corr = ticker_returns.corr(total_returns)
                    return corr

        def monthly_correlation_with_ticker(self, ticker):
            if ticker not in self.backtest.df.columns:
                raise ValueError(f"Ticker '{ticker}' not found in data.")
            
            # Ensure index is datetime
            if not isinstance(self.backtest.df.index, pd.DatetimeIndex):
                raise ValueError("DataFrame index must be a DatetimeIndex for monthly grouping.")
            
            # Calculate returns
            df = self.backtest.df.copy()
            df['ticker_returns'] = df[ticker].pct_change().fillna(0)
            df['total_returns'] = df['Total_equity'].pct_change().fillna(0)
            
            # Group by month and calculate correlation for each month
            monthly_corrs = df.groupby(pd.Grouper(freq='M')).apply(
                lambda x: x['ticker_returns'].corr(x['total_returns']) if len(x) >= 2 else np.nan
            )
            
            # Drop NaN months (e.g., months with insufficient data)
            monthly_corrs = monthly_corrs.dropna()
            
            if monthly_corrs.empty:
                return {'average': np.nan, 'max_corr': np.nan, 'max_month': None, 'min_corr': np.nan, 'min_month': None}
            
            # Calculate summary
            average = monthly_corrs.mean()
            max_corr = monthly_corrs.max()
            max_month = monthly_corrs.idxmax().strftime('%Y-%m')
            min_corr = monthly_corrs.min()
            min_month = monthly_corrs.idxmin().strftime('%Y-%m')
            
            return {
                'average': average,
                'max_corr': max_corr,
                'max_month': max_month,
                'min_corr': min_corr,
                'min_month': min_month
            }
        
        def profit_loss_ratio(self, ticker, trade_type='total'):
                    """
                    Calculate profit/loss ratio for total, long, or short trades: total profits / total losses (absolute value)
                    """
                    results = self.backtest.results.get(ticker, {'long': [], 'short': []})
                    if trade_type == 'long':
                        pnls = results['long']
                    elif trade_type == 'short':
                        pnls = results['short']
                    elif trade_type == 'total':
                        pnls = results['long'] + results['short']
                    else:
                        raise ValueError("trade_type must be 'total', 'long', or 'short'")
                    
                    if not pnls:
                        return np.nan
                    
                    profits = sum(p for p in pnls if p > 0)
                    losses = abs(sum(p for p in pnls if p < 0))
                    if losses == 0:
                        return np.inf if profits > 0 else np.nan
                    return profits / losses

        def average_monthly_trades(self, ticker, trade_type='total'):
            """
            Calculate average monthly number of trades for total, long, or short (based on close_signal)
            """
            if not isinstance(self.backtest.df.index, pd.DatetimeIndex):
                raise ValueError("DataFrame index must be a DatetimeIndex for monthly grouping.")
            
            df = self.backtest.df.copy()
            close_signal_col = f'{ticker}_close_signal'
            
            if trade_type == 'long':
                df['trade_count'] = (df[close_signal_col] == -1).astype(int)  # Close long: action_signal=-1
            elif trade_type == 'short':
                df['trade_count'] = (df[close_signal_col] == 1).astype(int)  # Close short: action_signal=1
            elif trade_type == 'total':
                df['trade_count'] = (df[close_signal_col] != 0).astype(int)
            else:
                raise ValueError("trade_type must be 'total', 'long', or 'short'")
            
            # Group by month and sum trades per month
            monthly_trades = df.groupby(pd.Grouper(freq='M'))['trade_count'].sum()
            
            # Optional: Exclude months with zero trades
            monthly_trades = monthly_trades[monthly_trades > 0]
            if monthly_trades.empty:
                return np.nan
            
            return monthly_trades.mean()