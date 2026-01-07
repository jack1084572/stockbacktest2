# =========================================================
# Imports
# =========================================================
import matplotlib
matplotlib.use("Agg")

import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime

# =========================================================
# Strategy: Linear Regression Trend using Daily Data
# =========================================================
class LinearRegressionStrategy(bt.Strategy):
    params = dict(
        period=20,                # 日线回归周期
        initial_shares=100,
        max_capital_usage_pct=0.9,
        stop_loss_cash=100,
        take_profit_cash=100,
        recovery_mult=2,
        daily_data=None           # 日线数据，用于趋势判断
    )

    def __init__(self):
        self.in_recovery = False
        self.recovery_shares = self.p.initial_shares
        self.last_trade_direction = None

        self.trade_log = []
        self.equity_curve = []

        self._prev_pos = 0
        self._entry = None

        if self.p.daily_data is None or len(self.p.daily_data) < self.p.period:
            raise ValueError("Daily data is required and must be longer than period")
        self.daily_close = self.p.daily_data["close"].values

    def linear_regression_slope_daily(self):
        # 取最近 period 根日线收盘
        y = self.daily_close[-self.p.period:]
        x = np.arange(len(y))
        A = np.vstack([x, np.ones(len(x))]).T
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return slope

    def check_capital(self, shares):
        price = self.data.close[0]
        return abs(price * shares) <= self.broker.getvalue() * self.p.max_capital_usage_pct

    def next(self):
        self.equity_curve.append(self.broker.getvalue())

        slope = self.linear_regression_slope_daily()

        if not self.position:
            shares = self.recovery_shares if self.in_recovery else self.p.initial_shares
            if not self.check_capital(shares):
                return

            if not self.in_recovery:
                if slope > 0:
                    self.last_trade_direction = "LONG"
                    self.buy(size=shares)
                elif slope < 0:
                    self.last_trade_direction = "SHORT"
                    self.sell(size=shares)
            else:
                # 复利回补
                if self.last_trade_direction == "LONG":
                    self.last_trade_direction = "SHORT"
                    self.sell(size=shares)
                else:
                    self.last_trade_direction = "LONG"
                    self.buy(size=shares)
        else:
            # 止损止盈
            entry_price = self.position.price
            price = self.data.close[0]
            shares = abs(self.position.size)

            pnl = (price - entry_price) * shares if self.position.size > 0 else (entry_price - price) * shares
            scale = shares / self.p.initial_shares
            if pnl <= -self.p.stop_loss_cash * scale:
                self.close()
            elif pnl >= self.p.take_profit_cash * scale:
                self.close()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            new_pos = self.position.size
            prev_pos = self._prev_pos
            price = order.executed.price
            dt = bt.num2date(order.executed.dt).strftime("%Y-%m-%d %H:%M")

            if prev_pos == 0 and new_pos != 0:
                self._entry = {"Entry Date": dt,
                               "Direction": "LONG" if new_pos > 0 else "SHORT",
                               "Shares": abs(new_pos),
                               "Entry Price": price}
            elif prev_pos != 0 and new_pos == 0 and self._entry:
                qty = self._entry["Shares"]
                pnl = ((price - self._entry["Entry Price"]) * qty
                       if self._entry["Direction"] == "LONG"
                       else (self._entry["Entry Price"] - price) * qty)

                if pnl < 0:
                    self.in_recovery = True
                    self.recovery_shares *= self.p.recovery_mult
                else:
                    self.in_recovery = False
                    self.recovery_shares = self.p.initial_shares

                self.trade_log.append({
                    "Entry Date": self._entry["Entry Date"],
                    "Exit Date": dt,
                    "Direction": self._entry["Direction"],
                    "Shares": qty,
                    "Entry Price": self._entry["Entry Price"],
                    "Exit Price": price,
                    "PnL ($)": round(pnl, 2)
                })
                self._entry = None

            self._prev_pos = new_pos

# =========================================================
# Data
# =========================================================
def download_and_save_data(symbol, period, interval, filename):
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })
    df["openinterest"] = 0
    df.to_csv(filename)
    print(f"Saved {interval} data: {filename}, rows: {len(df)}")
    return df

def load_csv_data(filename):
    df = pd.read_csv(filename, index_col=0, parse_dates=True)
    return df[["open","high","low","close","volume","openinterest"]]

# =========================================================
# HTML Report
# =========================================================
def generate_html_report(symbol, params, equity_curve, trades):
    total_pnl = sum(t["PnL ($)"] for t in trades)
    win_trades = sum(1 for t in trades if t["PnL ($)"] > 0)
    loss_trades = len(trades) - win_trades
    win_rate = (win_trades / len(trades) * 100) if trades else 0

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{symbol} Backtest Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body {{ font-family: Arial; margin: 40px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 6px; text-align: right; }}
th {{ background: #f4f4f4; }}
</style>
</head>

<body>
<h1>{symbol} Backtest Report</h1>

<h2>Parameters</h2>
<pre>{json.dumps(params, indent=2)}</pre>

<h2>Statistics</h2>
<ul>
<li>Total Trades: {len(trades)}</li>
<li>Winning Trades: {win_trades}</li>
<li>Losing Trades: {loss_trades}</li>
<li>Win Rate: {win_rate:.2f}%</li>
<li>Total PnL: ${round(total_pnl,2)}</li>
</ul>

<h2>Equity Curve</h2>
<canvas id="equityChart"></canvas>

<h2>Trade Log</h2>
<table>
<tr>
{''.join(f"<th>{k}</th>" for k in trades[0].keys()) if trades else ""}
</tr>
{''.join('<tr>' + ''.join(f"<td>{v}</td>" for v in t.values()) + '</tr>' for t in trades)}
</table>

<script>
const equityData = {json.dumps(equity_curve)};
new Chart(document.getElementById('equityChart'), {{
    type: 'line',
    data: {{
        labels: equityData.map((_, i) => i),
        datasets: [{{
            label: 'Equity',
            data: equityData,
            borderColor: 'blue',
            fill: false
        }}]
    }}
}});
</script>

</body>
</html>
"""
    with open(f"{symbol}.html", "w", encoding="utf-8") as f:
        f.write(html)

# =========================================================
# Main
# =========================================================
def main():
    stock_symbol = "SOXL"

    # 数据文件名
    csv_30m = f"{stock_symbol}_30m.csv"
    csv_1d = f"{stock_symbol}_1d.csv"

    # 下载最大免费数据并保存
    df_30m = download_and_save_data(stock_symbol, period="60d", interval="30m", filename=csv_30m)
    df_1d = download_and_save_data(stock_symbol, period="max", interval="1d", filename=csv_1d)

    # 加载回测数据
    df_30m = load_csv_data(csv_30m)
    df_1d = load_csv_data(csv_1d)

    data = bt.feeds.PandasData(
        dataname=df_30m,
        timeframe=bt.TimeFrame.Minutes,
        compression=30
    )

    cerebro = bt.Cerebro()
    cerebro.adddata(data)

    # 回测参数
    initial_shares = 100
    max_capital_usage_pct = 0.9
    initial_capital = 100000
    stop_loss_cash = initial_shares * 3
    take_profit_cash = stop_loss_cash
    recovery_mult = 2
    period_lr = 20

    cerebro.addstrategy(
        LinearRegressionStrategy,
        period=period_lr,
        initial_shares=initial_shares,
        max_capital_usage_pct=max_capital_usage_pct,
        stop_loss_cash=stop_loss_cash,
        take_profit_cash=take_profit_cash,
        recovery_mult=recovery_mult,
        daily_data=df_1d
    )

    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=0.001)

    strat = cerebro.run()[0]

    # 保存交易 CSV
    trades_df = pd.DataFrame(strat.trade_log)
    trades_csv_file = f"{stock_symbol}_trades.csv"
    trades_df.to_csv(trades_csv_file, index=False)
    print(f"Saved trades: {trades_csv_file}, trades: {len(trades_df)}")

    # 输出 HTML 报告
    generate_html_report(
        stock_symbol,
        params=dict(
            period=period_lr,
            initial_shares=initial_shares,
            stop_loss_cash=stop_loss_cash,
            take_profit_cash=take_profit_cash,
            recovery_mult=recovery_mult
        ),
        equity_curve=strat.equity_curve,
        trades=strat.trade_log
    )
    print(f"HTML report generated: {stock_symbol}.html")

if __name__ == "__main__":
    main()
