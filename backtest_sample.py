import pandas as pd
import pandas_ta as p_ta
import yfinance as yf
import numpy as np
import ta
from backtest_engine import framework 
from ta.volatility import AverageTrueRange
import talib
from vbp_calculator import vbp

class Strategy:
    def __init__(self, data, **kwargs):
        self.data = data
        self.short_window = 10
        self.long_window = 25
        self.test_window = kwargs.get('test_window', 1) 
        self.maintenance_margin_rate = kwargs.get('maintenance_margin_rate', 0.05)

    def add_reference_data(self, framework, ticker):
        framework.df['Maintenance_Margin'] = 0
        framework.df['force_close_out'] = 0
        framework.df['trend_strength'] = 0
        framework.df['trend'] = 0
        framework.df['trading_strategy'] = 0
        framework.df[f'{ticker}_best_price'] = 0.0
        framework.df[f'{ticker}_highest_high'] = 0.0
        framework.df[f'{ticker}_lowest_low'] = 0.0
        framework.df.index = pd.to_datetime(framework.df.index, utc=True)
        framework.df['hour'] = framework.df.index.hour 
        framework.df['weekday'] = framework.df.index.weekday
        framework.df['Mid'] = (framework.df['High'] + framework.df['Low'])/2
        framework.df['EMA_5'] = framework.df[ticker].ewm(span=5, adjust=False).mean()
        framework.df['EMA_20'] = framework.df[ticker].ewm(span=20, adjust=False).mean()
        framework.df['EMA_50'] = framework.df[ticker].ewm(span=50, adjust=False).mean()
        framework.df['EMA_100'] = framework.df[ticker].ewm(span=100, adjust=False).mean()
        framework.df['EMA_200'] = framework.df[ticker].ewm(span=200, adjust=False).mean()

        # Bollinger Band
        framework.df['Middle_Band'] = framework.df[ticker].ewm(span=20, adjust=False).mean()
        std = framework.df[ticker].rolling(window=20).std()
        framework.df['Upper_Band'] = framework.df['Middle_Band'] + (2 * std)
        framework.df['Lower_Band'] = framework.df['Middle_Band'] - (2 * std)
        framework.df['BB_width'] = (framework.df['Upper_Band'] - framework.df['Lower_Band']) / framework.df['Middle_Band']


        # RSI
        def compute_rsi(series, period=14):
            delta = series.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=period, min_periods=period).mean()
            avg_loss = loss.rolling(window=period, min_periods=period).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        framework.df['RSI'] = compute_rsi(framework.df[ticker], period=14)


        # ATR
        atr = AverageTrueRange(high=framework.df['High'], low=framework.df['Low'], close=framework.df[ticker], window=14)
        framework.df['ATR_14'] = atr.average_true_range()   

        # ADX
        adx_hourly= p_ta.adx(high=framework.df['High'], low=framework.df['Low'], close=framework.df[ticker], length=14)
        framework.df['ADX'] = adx_hourly['ADX_14']
        framework.df['+DI'] = adx_hourly['DMP_14']
        framework.df['-DI'] = adx_hourly['DMN_14']

        # spot and futures return
        framework.df['spot_return'] = framework.df['close_spot'].pct_change().fillna(0)  # 替换'spot'为实际spot价格列名
        framework.df['futures_return'] = framework.df[ticker].pct_change().fillna(0)




        # MACD
        close = framework.df[ticker]

        fast_macd_parameter = {1:12,2:8,3:5}
        slow_macd_parameter = {1:26,2:17,3:15}
        macd_singal_parameter = {1:9,2:9,3:5}

        #  12-period EMA and 26-period EMA
        ema_fast_12 = close.ewm(span=fast_macd_parameter[1], adjust=False).mean()
        ema_slow_26 = close.ewm(span=slow_macd_parameter[1], adjust=False).mean()

        # MACD line = ema_fast - ema_slow
        framework.df["MACD_12_26_9"] = ema_fast_12 - ema_slow_26

        # singal line = MACD  9-period EMA
        framework.df["MACD_signal_12_26_9"] = framework.df["MACD_12_26_9"].ewm(span=macd_singal_parameter[1], adjust=False).mean()

        # histogram = MACD - singal line
        framework.df["MACD_hist_12_26_9"] = framework.df["MACD_12_26_9"] - framework.df["MACD_signal_12_26_9"]
        

        #Drop NaN
        framework.df.dropna(inplace=True)

    #============================================================================================================================
    # Liquidation Check - Perpetual Futures (Bybit/Binance/OKX compatible)
    #============================================================================================================================
    def check_liquidation(self, framework, ticker, index):
        """
        Check if the current bar's High/Low has triggered liquidation.
        Uses the exact leverage that was applied when the position was opened.
        """
        pos = framework.df.loc[index, ticker + '_holding_position']
        if pos == 0:
            return False, framework.df.loc[index, ticker]   # No position → no liquidation

        # Retrieve the leverage used when this position was originally opened
        open_leverage = framework.df.loc[index, ticker + '_open_leverage']
        if pd.isna(open_leverage) or open_leverage <= 1:
            open_leverage = 5   # Fallback for legacy positions

        avg_price = framework.df.loc[index, ticker + '_average_cost']
        mmr       = self.maintenance_margin_rate          # Maintenance Margin Rate (e.g. 0.005 = 0.5%)

        if pos > 0:  # === Long position ===
            # Official simplified liquidation price formula (2025)
            liq_price = avg_price * (1 - (1 - mmr) / open_leverage)
            
            if framework.df.loc[index, 'Low'] <= liq_price:   # Wicked down to or below liquidation price
                return True, liq_price

        elif pos < 0:  # === Short position ===
            liq_price = avg_price * (1 + (1 - mmr) / open_leverage)
            
            if framework.df.loc[index, 'High'] >= liq_price:  # Wicked up to or above liquidation price
                return True, liq_price

        return False, framework.df.loc[index, ticker]   # No liquidation this bar
    

#============================================================================================================================
# Next
#============================================================================================================================

    def next(self, ticker, framework, index):
        current_index = framework.df.index.get_loc(index)
        if current_index < 2:  #ignore first index 
            return

        last_index = framework.df.index[current_index - 1]
        last2_index = framework.df.index[current_index - 2]
        last3_index = framework.df.index[current_index - 3]
        last4_index = framework.df.index[current_index - 4]
        last5_index = framework.df.index[current_index - 5]
        last6_index = framework.df.index[current_index - 6]
        last9_index = framework.df.index[current_index - 9]
        last10_index = framework.df.index[current_index - 10]
        last12_index = framework.df.index[current_index - 12]
        last13_index = framework.df.index[current_index - 13]
        last14_index = framework.df.index[current_index - 14]
        last15_index = framework.df.index[current_index - 15]
        last17_index = framework.df.index[current_index - 17]
        last19_index = framework.df.index[current_index - 19]
        last20_index = framework.df.index[current_index - 20]
        last21_index = framework.df.index[current_index - 21]
        last23_index = framework.df.index[current_index - 23]
        last25_index = framework.df.index[current_index - 25]
        last49_index = framework.df.index[current_index - 49]
        last73_index = framework.df.index[current_index - 73]
        last97_index = framework.df.index[current_index - 97]
        last121_index = framework.df.index[current_index - 121]

        
        # periouvs data
        framework.df.loc[index, 'Maintenance_Margin'] = framework.df.loc[last_index, 'Maintenance_Margin']
        framework.df.loc[index, ticker + '_holding_position'] = framework.df.loc[last_index, ticker + '_holding_position']
        framework.df.loc[index, ticker + '_initial_margin'] = framework.df.loc[last_index, ticker + '_initial_margin']
        framework.df.loc[index, ticker + '_open_leverage'] = framework.df.loc[last_index, ticker + '_open_leverage']
        framework.df.loc[index, ticker + '_long_trade'] = framework.df.loc[last_index, ticker + '_long_trade']
        framework.df.loc[index, ticker + '_long_win'] = framework.df.loc[last_index, ticker + '_long_win']
        framework.df.loc[index, ticker + '_long_win_rate'] = framework.df.loc[last_index, ticker + '_long_win_rate']
        framework.df.loc[index, ticker + '_short_trade'] = framework.df.loc[last_index, ticker + '_short_trade']
        framework.df.loc[index, ticker + '_short_win'] = framework.df.loc[last_index, ticker + '_short_win']
        framework.df.loc[index, ticker + '_short_win_rate'] = framework.df.loc[last_index, ticker + '_short_win_rate']
        if framework.df.loc[last_index, ticker + '_holding_position'] != 0:
            framework.df.loc[index, ticker + '_average_cost'] = framework.df.loc[last_index, ticker + '_average_cost']
        else:
            framework.df.loc[index, ticker + '_average_cost'] = 0
        framework.df.loc[index, 'Cash'] = framework.df.loc[last_index, 'Cash']
        framework.df.loc[index, 'Total_equity'] = framework.df.loc[last_index, 'Total_equity']
        if framework.df.loc[index, ticker + '_holding_position'] != 0:
            framework.df.loc[index, 'trading_strategy'] = framework.df.loc[last_index, 'trading_strategy']
        best_price = framework.df.loc[last_index, f'{ticker}_best_price']
        highest_high = framework.df.loc[last_index, f'{ticker}_highest_high']
        lowest_low = framework.df.loc[last_index, f'{ticker}_lowest_low']

        if framework.df.loc[index, ticker + '_holding_position'] != 0:
                    current_price = framework.df.loc[index, ticker]
                    current_high = framework.df.loc[index, 'High']
                    current_low = framework.df.loc[index, 'Low']
                    if framework.df.loc[index, ticker + '_holding_position'] > 0:  
                        best_price = max(best_price, current_price)
                        highest_high = max(highest_high, current_high)
                        
                    elif framework.df.loc[index, ticker + '_holding_position'] < 0: 
                        best_price = min(best_price, current_price)
                        lowest_low = min(lowest_low, current_low)
                    
                    framework.df.loc[index, f'{ticker}_best_price'] = best_price
                    framework.df.loc[index, f'{ticker}_highest_high'] = highest_high
                    framework.df.loc[index, f'{ticker}_lowest_low'] = lowest_low



        is_liq, exec_price = self.check_liquidation(framework, ticker, index)
        if is_liq:
            holding = abs(framework.df.loc[index, ticker + '_holding_position'])
            action = -1 if framework.df.loc[index, ticker + '_holding_position'] > 0 else 1
            
            framework.close_position(index, ticker, exec_price, holding,
                                action_signal=action, strategy_type='LIQUIDATED')
            
            framework.df.loc[index, 'force_close_out'] = 1
            # Liquidation fee
            framework.df.loc[index, 'Total_equity'] = max(10, framework.df.loc[index, 'Total_equity'] * 0.05)
            
            # Update position info after liquidation
            framework.update_position_info(index, ticker)
            return

        

    
        adx = framework.df.loc[last_index, 'ADX']
        atr = framework.df.loc[last_index, 'ATR_14']

        bb_width = framework.df.loc[last_index, 'BB_width']

        num = 10

        num -= int(atr/100)


        leverage = 5
        
        # current market data
        total_equity = framework.df.loc[last_index, 'Total_equity']
        open = framework.df.loc[last_index, 'Open']
        price = framework.df.loc[last_index, ticker]
        last_price = framework.df.loc[last2_index, ticker]
        prev_5_price = framework.df.loc[last4_index:last_index, ticker]
        high = framework.df.loc[last_index, 'High']
        last_high = framework.df.loc[last2_index, 'High']
        prev_5_high = framework.df.loc[last4_index:last_index, 'High']
        low = framework.df.loc[last_index, 'Low']
        prev_5_low = framework.df.loc[last4_index:last_index, 'Low']
        last_low = framework.df.loc[last2_index, 'Low']
        prev_10_price = framework.df.loc[last9_index:last_index, ticker]
        prev_15_price = framework.df.loc[last14_index:last_index, ticker]
        current_open = framework.df.loc[index, 'Open']
        
        # current position data
        best_price = framework.df.loc[last_index, f'{ticker}_best_price']
        margin_per_contract = current_open/leverage
        average_cost = framework.df.loc[last_index, ticker + '_average_cost']
        strategy = framework.df.loc[last_index, 'trading_strategy']
        holding_position = framework.df.loc[last_index, ticker + '_holding_position']
        holding_period = framework.df.loc[last_index, ticker + '_holding_period']

        # TA indicator
        rsi = framework.df.loc[last_index, 'RSI']
        macd_12_26_9 = framework.df.loc[last_index, 'MACD_12_26_9']
        last_macd_12_26_9 = framework.df.loc[last2_index, 'MACD_12_26_9']
        macd_signal_12_26_9 = framework.df.loc[last_index, 'MACD_signal_12_26_9']
        last_macd_signal_12_26_9 = framework.df.loc[last2_index, 'MACD_signal_12_26_9']
        macd_hist_12_26_9 = framework.df.loc[last_index, 'MACD_hist_12_26_9']





        # condition 
        #signal
        macd_12_26_9_up_cross = macd_12_26_9 > macd_signal_12_26_9 and last_macd_12_26_9 < last_macd_signal_12_26_9
        macd_12_26_9_down_cross = macd_12_26_9 < macd_signal_12_26_9 and last_macd_12_26_9 > last_macd_signal_12_26_9

        

        # filter


        # trade quantity 
        buy_qty = int((total_equity*0.7)/margin_per_contract)

        #Trading condition
        #Open  
        basic_open_condition = framework.df.loc[last_index, 'Cash'] > margin_per_contract*(buy_qty+abs(framework.df.loc[last_index, ticker + '_holding_position'])) and buy_qty != False
        #long 


        # short 
        basic_close_condition = framework.df.loc[last_index, ticker + '_holding_position'] != 0


        # Close 

        # stop profit or loss
        stop_profit_condition = False
        stop_loss_condition = False

        stop_profit_parameter ={ 0:2,1:1.5,2:1.2}
        stop_loss_parameter = {0:1,1:1, 2:0.75}

        stop_profit_thersold = atr*stop_profit_parameter[1]
        stop_loss_thersold = atr*stop_loss_parameter[1]

        stop_point_multiplier = 1

        stop_long_profit_price = average_cost + (stop_profit_thersold * stop_point_multiplier)
        stop_long_loss_price = average_cost - (stop_loss_thersold*stop_point_multiplier ) 
        stop_short_profit_price = average_cost  - (stop_profit_thersold * stop_point_multiplier)
        stop_short_loss_price = average_cost  + (stop_loss_thersold*stop_point_multiplier )

        if framework.df.loc[last_index, ticker + '_holding_position'] > 0:
            stop_profit_condition = last_price >= stop_long_profit_price 
            stop_loss_condition = last_price <= stop_long_loss_price
        elif framework.df.loc[last_index, ticker + '_holding_position'] < 0:
            stop_profit_condition = last_price <= stop_short_profit_price
            stop_loss_condition = last_price >= stop_short_loss_price

        #Trend identify
        # strength



        if adx > 25 and adx < 40:
            framework.df.loc[index, 'trend_strength'] = 1
        elif adx > 40:
            framework.df.loc[index, 'trend_strength'] = 2



        #long singal
        open_long_con = {1: rsi < 30,
                         2: rsi > 50,
                         3: macd_12_26_9_up_cross
                         }


                        
        

        close_long_con = {1: rsi > 70,
                          2: rsi < 50,
                          3: macd_12_26_9_down_cross,
                          }

        


        #short singal
        open_short_con = {
                          1: macd_12_26_9_down_cross , 

                          }

        
        close_short_con = {
                           1: macd_12_26_9_up_cross < 0
                           }
    

       #==========================================================================================================
        #Trade
        #==========================================================================================================

        # close position

        closed_positon = False

        trade_price = current_open

        strategy_type = {1: 'Classic RSI ',
                        2: 'Sentiment-driven RSI',
                        3: 'Classic MACD'}

        if basic_close_condition and closed_positon == False:
            if framework.df.loc[last_index, ticker + '_holding_position'] > 0: 

                if (strategy == strategy_type[self.test_window] )  and close_long_con[self.test_window] :
                    framework.close_position(index, ticker, trade_price, holding_position, action_signal=-1, strategy_type = 'kdj_long_close')
                    closed_positon = True



            if framework.df.loc[last_index, ticker + '_holding_position'] < 0: 
                if (strategy == 'short')  and (close_short_con [1]) :  
                    framework.close_position(index, ticker, trade_price, holding_position, action_signal=1, strategy_type = 'close short position')
                    closed_positon = True


        if stop_profit_condition and closed_positon == False:
            if framework.df.loc[last_index, ticker + '_holding_position'] > 0 : 
                framework.close_position(index, ticker, trade_price, holding_position, action_signal=-1, strategy_type = 'stop profit')
                closed_positon = True

            if framework.df.loc[last_index, ticker + '_holding_position'] < 0 : 
                framework.close_position(index, ticker, trade_price, holding_position, action_signal=1, strategy_type = 'stop profit')
                closed_positon = True

                
        
        if stop_loss_condition and closed_positon == False:
            if framework.df.loc[last_index, ticker + '_holding_position'] > 0: 
                framework.close_position(index, ticker, trade_price, holding_position, action_signal=-1, strategy_type = 'stop loss')
                closed_positon = True

            if framework.df.loc[last_index, ticker + '_holding_position'] < 0: 
                framework.close_position(index, ticker, trade_price, holding_position, action_signal=1, strategy_type = 'stop loss')
                closed_positon = True


        #open 
        if basic_open_condition  and framework.df.loc[last_index, ticker + '_close_signal'] == 0 : 
            # Long  

            if framework.df.loc[last_index, ticker + '_holding_position'] < 1:
                
                
                if framework.df.loc[last_index, 'trend_strength'] < 3:
                    if  open_long_con[self.test_window] :
                        framework.open_position(index, ticker, trade_price, high, low, buy_qty, direction='long', strategy_type = strategy_type[self.test_window], leverage = leverage)

            # short
            if framework.df.loc[last_index, ticker + '_holding_position'] < 1:
                    if open_short_con[1] and None: # long only 
                        framework.open_position(index, ticker, trade_price, high, low, buy_qty, direction='short', strategy_type = 'short', leverage = leverage)









        #==========================================================================================================
        #Equity & cash moving
        #==========================================================================================================        
        #Maintenance Margin
        framework.df.loc[index, 'Maintenance_Margin'] = (framework.df.loc[index, ticker + '_holding_position']* framework.df.loc[index, ticker + '_average_cost'] *self.maintenance_margin_rate ) 

        
        framework.update_position_info(index, ticker)

        #Equity
        framework.df.loc[index, 'Total_equity'] = framework.df.loc[index, 'Cash'] + framework.df.loc[index, ticker +'_holding_PnL'] + framework.df.loc[index, ticker + '_initial_margin']


best_sharpe = float('-inf')


ratios = np.arange(1,4,1)

csv_file = 'eth_merged_data.csv'
start_date = '2024-11-28 00:00:00'
end_date = '2025-08-31 23:00:00'


equity_dict = {}  # To collect Total_equity series from each iteration
open_signal_dict = {}  # To collect open_signal series from each iteration
close_signal_dict = {}  # To collect close_signal series from each iteration
holding_pnl_dict = {}  # To collect holding_PnL series from each iteration  # ADDED HERE
price_series = None  # To store the price column (same for all)
atr_series = None  # To store the ATR_14 column (same for all, for visualization)


for i in ratios:

    ticker = 'ETH'
    
    stock_df = pd.read_csv(csv_file, parse_dates=['Date'])
    stock_df.set_index('Date', inplace=True)  
    stock_df.index = pd.to_datetime(stock_df.index, utc=True)
    filtered_df = stock_df.loc[start_date:end_date]
    
    fw = framework(initial_cash=10000)
    fw.add_data(ticker, filtered_df)
    fw.add_strategy(Strategy, test_window=i)
    fw.run()
    fw.calculate_return()
    
    sharpe = fw.analyse_tool.sharpe_ratio(periods = len(filtered_df))
    max_drawdown = fw.analyse_tool.maximum_drawdown()
    corr = fw.analyse_tool.correlation_with_ticker(ticker)
    monthly_corr = fw.analyse_tool.monthly_correlation_with_ticker(ticker)
    total_win_rate = (fw.df[f'{ticker}_long_win'].iloc[-1] + fw.df[f'{ticker}_short_win'].iloc[-1]) / (fw.df[f'{ticker}_long_trade'].iloc[-1] + fw.df[f'{ticker}_short_trade'].iloc[-1])
    pl_ratio = fw.analyse_tool.profit_loss_ratio(ticker)
    avg_monthly_trades = fw.analyse_tool.average_monthly_trades(ticker)

    
    print(f'Stock:{ticker}')
    print(f'window:{i}')
    print(f'SR:{sharpe}')
    print(f'MMD:{max_drawdown}')
    print(f"Total Profit/Loss Ratio: {pl_ratio}")
    print(f"long trades: {fw.df[f'{ticker}_long_trade'].iloc[-1]}")
    print(f"Average Monthly Trades: {avg_monthly_trades}")
    print(f"long win rate: {(fw.df[f'{ticker}_long_win_rate'].iloc[-1])*100}%")
    print(f"short trades: {fw.df[f'{ticker}_short_trade'].iloc[-1]}")
    print(f"short win rate: {(fw.df[f'{ticker}_short_win_rate'].iloc[-1])*100}%")
    print(f"total win rate: {total_win_rate*100}%")
    print(f"Overall correlation: {corr:.4f}")
    print(f"Average monthly correlation: {monthly_corr['average']:.4f}")
    print(f"Max correlation: {monthly_corr['max_corr']:.4f} in {monthly_corr['max_month']}")
    print(f"Min correlation: {monthly_corr['min_corr']:.4f} in {monthly_corr['min_month']}")
    
    if sharpe >= best_sharpe:
        best_sharpe = sharpe
        best_stock = ticker
        best_ratio = i
        best_mmd = max_drawdown
        optimized_df = fw.df
        best_win_rate = total_win_rate
        overall_correlation = corr
        average_monthly_correlation = monthly_corr['average']
        max_corr = monthly_corr['max_corr']
        min_corr = monthly_corr['min_corr']
        max_corr_month = monthly_corr['max_month']
        min_corr_month = monthly_corr['min_month']
        best_pl_ratio = pl_ratio
        best_avg_monthly_trades = avg_monthly_trades
    
    # Collect Total_equity for this parameter
    equity_dict[i] = fw.df['Total_equity'].copy()
    
    # Collect signals for this parameter
    open_signal_dict[i] = fw.df[f'{ticker}_open_signal'].copy()
    close_signal_dict[i] = fw.df[f'{ticker}_close_signal'].copy()
    holding_pnl_dict[i] = fw.df[f'{ticker}_holding_PnL'].copy()  # ADDED HERE
    
    # Collect price and ATR only once (same for all iterations)
    if price_series is None:
        price_series = fw.df[ticker].copy()
        atr_series = fw.df['ATR_14'].copy()

# After the loop, combine into one DataFrame with columns named by parameter (test_window)
combined_equity_df = pd.DataFrame(equity_dict)
combined_equity_df.columns = [f'Total_equity_window_{col}' for col in combined_equity_df.columns]

# Add the signal columns per window
for i in ratios:
    combined_equity_df[f'{ticker}_open_signal_window_{i}'] = open_signal_dict[i]
    combined_equity_df[f'{ticker}_close_signal_window_{i}'] = close_signal_dict[i]
    combined_equity_df[f'{ticker}_holding_PnL_window_{i}'] = holding_pnl_dict[i]  # ADDED HERE

# Add the common price and ATR columns
combined_equity_df[ticker] = price_series
combined_equity_df['ATR'] = atr_series  # CHANGED HERE from 'ATR_14'


combined_equity_df.to_csv('backtest_equity_comparison.csv', index_label="Date")

if best_sharpe:
    print(f'Stock:{best_stock}')
    print(f'Volume window:{best_ratio}')
    print(f'SR:{best_sharpe}')
    print(f'MMD:{best_mmd}')
    print(f"Profit/Loss Ratio: {best_pl_ratio}")
    print(f"Average Monthly Trades: {best_avg_monthly_trades}")
    print(f"long trades: {optimized_df[f'{ticker}_long_trade'].iloc[-1]}")
    print(f"long win rate: {(optimized_df[f'{ticker}_long_win_rate'].iloc[-1])*100}%")
    print(f"short trades: {optimized_df[f'{ticker}_short_trade'].iloc[-1]}")
    print(f"short win rate: {(optimized_df[f'{ticker}_short_win_rate'].iloc[-1])*100}%")
    print(f"total win rate: {best_win_rate*100}%")
    print(f"Overall correlation: {overall_correlation:.4f}")
    print(f"Average monthly correlation: {average_monthly_correlation:.4f}")
    print(f"Max correlation: {max_corr:.4f} in {max_corr_month}")
    print(f"Min correlation: {min_corr:.4f} in {min_corr_month}")

        

optimized_df.to_csv('backtest_result.csv', index_label="Date")

optimized_df.index = pd.to_datetime(optimized_df.index, format="%Y-%m-%d %H:%M:%S", errors='coerce')

if optimized_df.index.isna().any():
    print("Warning: Some index values could not be converted to datetime. Check the data:")
    print(optimized_df.index[optimized_df.index.isna()])


if optimized_df.index.isna().any():
    print("Warning: Some index values could not be converted to datetime. Check the data:")
    print(optimized_df.index[optimized_df.index.isna()])
