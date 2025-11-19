# Crypto Quant Engine – Production-Grade Event-Driven Backtesting Framework

A from-scratch, event-driven backtesting engine built and battle-tested with real Binance perpetual futures data (ETHUSDT hourly, Nov 2024 – Aug 2025).  
Designed with production details in mind: slippage, fees, maintenance margin, forced liquidation, dynamic ATR stops, position sizing, etc.

### Real Performance – Three Classic Strategies Head-to-Head

| Strategy                        | Total Return | Annualized | Sharpe | Max Drawdown | Trades | Win Rate |
|---------------------------------|--------------|------------|--------|--------------|--------|----------|
| Classic RSI (Long when RSI < 30) | -87%         | —          | -1.56  | -92%         | 157    | 45.9%    |
| Sentiment-driven RSI (Long when RSI > 50) | +28% | ~38%       | 0.24   | -76%         | 571    | 38.0%    |
| Classic MACD (12-26-9 crossover) | +30%         | ~41%       | 0.25   | -60%         | 243    | 40.7%    |

**Key takeaway in the 2024-2025 bull market: momentum/chasing (RSI > 50) dramatically outperforms traditional mean-reversion buying-the-dip.**

### Interactive Dashboard (instant local launch)

```bash
python equity_curve_dashboard.py
