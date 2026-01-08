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
import os
from datetime import datetime, timedelta

# =========================================================
# Strategy: Linear Regression Trend using Daily Data
# =========================================================
class LinearRegressionStrategy(bt.Strategy):
    params = dict(
        period=20,
        initial_shares=100,
        max_capital_usage_pct=0.9,
        recovery_mult=2,
        daily_data=None,
        take_profit_price=0.5,  # 固定价格止盈
        stop_loss_price=0.5     # 固定价格止损
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
                # 回补逻辑
                if self.last_trade_direction == "LONG":
                    self.last_trade_direction = "SHORT"
                    self.sell(size=shares)
                else:
                    self.last_trade_direction = "LONG"
                    self.buy(size=shares)
        else:
            entry_price = self.position.price
            price = self.data.close[0]
            shares = abs(self.position.size)

            # 使用固定价格止盈止损
            if self.position.size > 0:  # 多单
                if price >= entry_price + self.p.take_profit_price or price <= entry_price - self.p.stop_loss_price:
                    self.close()
            else:  # 空单
                if price <= entry_price - self.p.take_profit_price or price >= entry_price + self.p.stop_loss_price:
                    self.close()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            new_pos = self.position.size
            prev_pos = self._prev_pos
            price = order.executed.price
            dt = bt.num2date(order.executed.dt).strftime("%Y-%m-%d %H:%M")
            size = order.executed.size

            # 开仓或加仓
            if prev_pos == 0 and new_pos != 0:
                self._entry = {
                    "Entry Date": dt,
                    "Direction": "LONG" if new_pos > 0 else "SHORT",
                    "Shares": abs(size),
                    "Entry Price": price
                }
            elif prev_pos != 0 and new_pos != 0:
                # 加仓，更新平均价格
                total_shares = self._entry["Shares"] + abs(size)
                avg_price = (self._entry["Entry Price"] * self._entry["Shares"] + price * abs(size)) / total_shares
                self._entry["Entry Price"] = avg_price
                self._entry["Shares"] = total_shares

            # 平仓
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

                balance = self.broker.getvalue()  # 平仓时账户余额

                self.trade_log.append({
                    "Entry Date": self._entry["Entry Date"],
                    "Exit Date": dt,
                    "Direction": self._entry["Direction"],
                    "Shares": qty,
                    "Entry Price": round(self._entry["Entry Price"], 4),
                    "Exit Price": round(price, 4),
                    "PnL ($)": round(pnl, 2),
                    "Equity ($)": round(balance, 2)
                })

                self._entry = None

            self._prev_pos = new_pos

# =========================================================
# Data Utilities
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
    return df[["open", "high", "low", "close", "volume", "openinterest"]]

# =========================================================
# Simulate 30m from Daily
# =========================================================
def simulate_30m_from_daily(daily_df, filename):
    rows = []
    for dt, row in daily_df.iterrows():
        day_open, day_close, day_high, day_low, volume = row[["open","close","high","low","volume"]]
        times = pd.date_range(start=dt.replace(hour=9, minute=30), periods=13, freq="30min")
        base = np.linspace(day_open, day_close, 13)
        noise = np.random.normal(0, (day_high - day_low) * 0.05, 13)
        prices = np.clip(base + noise, day_low, day_high)
        prices[0] = day_open
        prices[-1] = day_close

        for i, t in enumerate(times):
            o = prices[i]
            c = prices[i]
            h = min(max(o,c) + abs(noise[i])*0.3, day_high)
            l = max(min(o,c) - abs(noise[i])*0.3, day_low)
            rows.append({
                "datetime": t,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": volume/13,
                "openinterest": 0
            })

    df_30m = pd.DataFrame(rows).set_index("datetime")
    df_30m.to_csv(filename)
    print(f"Simulated 30m data saved: {filename}, rows: {len(df_30m)}")
    return df_30m

# =========================================================
# HTML Report
# =========================================================
def generate_html_report(symbol, params, equity_curve, trades):
    total_pnl = sum(t["PnL ($)"] for t in trades)
    win_trades = sum(1 for t in trades if t["PnL ($)"] > 0)
    loss_trades = len(trades) - win_trades
    win_rate = (win_trades / len(trades) * 100) if trades else 0

    trade_table_header = ""
    if trades:
        trade_table_header = "<tr>" + "".join(f"<th>{k}</th>" for k in trades[0].keys()) + "</tr>"

    trade_table_rows = ""
    for t in trades:
        trade_table_rows += "<tr>" + "".join(f"<td>{v}</td>" for v in t.values()) + "</tr>"

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{symbol} Backtest Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body {{ font-family: Arial; margin: 40px; }}
table {{ border-collapse: collapse; width: 100%; margin-top:20px; }}
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
{trade_table_header}
{trade_table_rows}
</table>

<script>
const equityData = {json.dumps(equity_curve)};
new Chart(document.getElementById('equityChart'), {{
    type: 'line',
    data: {{
        labels: equityData.map((_, i) => i),
        datasets: [{{ label: 'Equity', data: equityData, borderColor: 'blue', fill: false }}]
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
    symbol = "SOXL"
    backtest_years = 1  # 回测范围：最近几年，1 表示 1 年，2 表示 2 年

    csv_1d = f"{symbol}_1d.csv"
    csv_30m = f"{symbol}_30m.csv"

    # 1. 下载或读取 CSV
    if not os.path.exists(csv_1d):
        df_1d = download_and_save_data(symbol, "max", "1d", csv_1d)
    else:
        print(f"Using existing {csv_1d}")
        df_1d = load_csv_data(csv_1d)

    if not os.path.exists(csv_30m):
        simulate_30m_from_daily(df_1d, csv_30m)
    else:
        print(f"Using existing {csv_30m}")

    df_1d = load_csv_data(csv_1d)
    df_30m = load_csv_data(csv_30m)

    # 过滤回测范围
    start_date = datetime.now() - timedelta(days=365 * backtest_years)
    df_1d = df_1d[df_1d.index >= start_date]
    df_30m = df_30m[df_30m.index >= start_date]

    # 2. 回测
    data = bt.feeds.PandasData(dataname=df_30m, timeframe=bt.TimeFrame.Minutes, compression=30)
    cerebro = bt.Cerebro()
    cerebro.adddata(data)

    strategy_params = dict(
        period=20,
        initial_shares=100,
        recovery_mult=2,
        max_capital_usage_pct=0.9,
        take_profit_price=4,  # 固定价格止盈
        stop_loss_price=4      # 固定价格止损
    )

    cerebro.addstrategy(LinearRegressionStrategy, daily_data=df_1d, **strategy_params)
    cerebro.broker.setcash(100000)
    cerebro.broker.setcommission(commission=0.001)

    strat = cerebro.run()[0]

    # 3. 保存交易 CSV
    pd.DataFrame(strat.trade_log).to_csv(f"{symbol}_trades.csv", index=False)
    print(f"Saved trades: {symbol}_trades.csv, total trades: {len(strat.trade_log)}")

    # 4. 输出 HTML 报告
    generate_html_report(symbol, strategy_params, strat.equity_curve, strat.trade_log)
    print(f"HTML report generated: {symbol}.html")


if __name__ == "__main__":
    main()
