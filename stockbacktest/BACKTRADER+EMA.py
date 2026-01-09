# =========================================================
# Imports
# =========================================================
import os
import json
import numpy as np
import pandas as pd
import backtrader as bt
import yfinance as yf
from datetime import timedelta

# =========================================================
# Strategy: EMA Compare + Recovery Scaling
# =========================================================
class EMAStrategy(bt.Strategy):
    params = dict(
        fast_period=9,
        slow_period=21,
        initial_shares=100,
        stop_loss_cash=300,
        take_profit_cash=300,
        recovery_mult=2,
        max_capital_pct=0.9,
    )

    def __init__(self):
        self.ema_fast = bt.ind.EMA(self.data.close, period=self.p.fast_period)
        self.ema_slow = bt.ind.EMA(self.data.close, period=self.p.slow_period)

        self.trade_log = []
        self.equity_curve = []

        self.in_recovery = False
        self.recovery_shares = self.p.initial_shares

        self._prev_pos = 0
        self._entry = None

    def check_capital(self, shares):
        price = self.data.close[0]
        return abs(price * shares) <= self.broker.getvalue() * self.p.max_capital_pct

    def next(self):
        self.equity_curve.append(round(self.broker.getvalue(), 2))

        fast = self.ema_fast[0]
        slow = self.ema_slow[0]
        price = self.data.close[0]

        if not self.position:
            shares = self.recovery_shares if self.in_recovery else self.p.initial_shares
            if not self.check_capital(shares):
                return
            if fast > slow:
                self.buy(size=shares)
            elif fast < slow:
                self.sell(size=shares)
        else:
            entry_price = self.position.price
            shares = abs(self.position.size)
            pnl = (
                (price - entry_price) * shares
                if self.position.size > 0
                else (entry_price - price) * shares
            )
            scale = shares / self.p.initial_shares
            if pnl <= -self.p.stop_loss_cash * scale or pnl >= self.p.take_profit_cash * scale:
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
                self._entry = {
                    "Entry Date": dt,
                    "Direction": "LONG" if new_pos > 0 else "SHORT",
                    "Shares": abs(new_pos),
                    "Entry Price": round(price, 2),
                }

            elif prev_pos != 0 and new_pos == 0 and self._entry:
                qty = self._entry["Shares"]
                pnl = (
                    (price - self._entry["Entry Price"]) * qty
                    if self._entry["Direction"] == "LONG"
                    else (self._entry["Entry Price"] - price) * qty
                )

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
                    "Exit Price": round(price, 2),
                    "PnL ($)": round(pnl, 2),
                    "Equity After Close": round(self.broker.getvalue(), 2),
                })
                self._entry = None

            self._prev_pos = new_pos

# =========================================================
# Data Utilities
# =========================================================
def generate_random_30m_from_daily(daily_df, out_csv):
    rows = []
    for date, row in daily_df.iterrows():
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        base = np.linspace(o, c, 13)
        noise = np.random.normal(0, (h - l) * 0.05, size=13)
        prices = base + noise
        t = pd.Timestamp(date)
        for i in range(13):
            rows.append({
                "datetime": t,
                "open": prices[i],
                "high": prices[i] + abs(noise[i]),
                "low": prices[i] - abs(noise[i]),
                "close": prices[i],
                "volume": row["volume"] / 13,
                "openinterest": 0,
            })
            t += timedelta(minutes=30)

    df30 = pd.DataFrame(rows).set_index("datetime")
    df30.to_csv(out_csv)
    return df30

def prepare_data(symbol, months=120):
    """
    1. 检查是否存在日线 CSV，如果没有下载
    2. 截取最近 months 月数据生成30分钟 CSV
    3. 返回30分钟 DataFrame
    """
    daily_csv = f"{symbol}_1d.csv"
    m30_csv = f"{symbol}_30m.csv"

    # 下载日线数据
    if not os.path.exists(daily_csv):
        print(f"Downloading full daily data for {symbol}...")
        df = yf.download(symbol, period="max", interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)
        df["openinterest"] = 0
        df.to_csv(daily_csv)
        print(f"Saved daily CSV: {daily_csv}")
    else:
        df = pd.read_csv(daily_csv, index_col=0, parse_dates=True)

    # 截取最近 months 月数据
    last_date = df.index.max()
    start_date = last_date - pd.DateOffset(months=months)
    df_recent = df[df.index >= start_date]

    # 生成30分钟 CSV
    print(f"Generating 30m data from last {months} months...")
    df30 = generate_random_30m_from_daily(df_recent, m30_csv)
    print(f"Saved 30m CSV: {m30_csv}")

    return df30[["open", "high", "low", "close", "volume", "openinterest"]]

# =========================================================
# HTML Report
# =========================================================
def generate_html(symbol, params, equity_curve, trades):
    total_trades = len(trades)
    wins = sum(1 for t in trades if t["PnL ($)"] > 0)
    losses = total_trades - wins
    total_pnl = round(sum(t["PnL ($)"] for t in trades), 2)
    win_rate = (wins / total_trades * 100) if total_trades else 0

    params_html = "".join(f"<li>{k}: {v}</li>" for k, v in params.items())
    equity_json = json.dumps(equity_curve)

    trade_rows = ""
    for t in trades:
        direction_color = "green" if t["Direction"] == "LONG" else "red"
        pnl_color = "green" if t["PnL ($)"] > 0 else "red"
        trade_rows += "<tr>"
        trade_rows += f"<td>{t['Entry Date']}</td>"
        trade_rows += f"<td>{t['Exit Date']}</td>"
        trade_rows += f"<td style='color:{direction_color}; font-weight:bold'>{t['Direction']}</td>"
        trade_rows += f"<td>{t['Shares']}</td>"
        trade_rows += f"<td>{t['Entry Price']}</td>"
        trade_rows += f"<td>{t['Exit Price']}</td>"
        trade_rows += f"<td style='color:{pnl_color}; font-weight:bold'>{t['PnL ($)']}</td>"
        trade_rows += f"<td>{t['Equity After Close']}</td>"
        trade_rows += "</tr>\n"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{symbol} Backtest Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body {{ font-family: Arial, sans-serif; margin: 30px; }}
h1 {{ border-bottom: 2px solid #333; padding-bottom: 5px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 15px; }}
th, td {{ border: 1px solid #ccc; padding: 6px; text-align: center; }}
th {{ background: #f2f2f2; }}
</style>
</head>
<body>

<h1>{symbol} Backtest Report</h1>

<h2>Strategy Parameters</h2>
<ul>{params_html}</ul>

<h2>Statistics</h2>
<ul>
<li>Total Trades: {total_trades}</li>
<li>Win Rate: {win_rate:.2f}%</li>
<li>Total PnL: {total_pnl}</li>
<li>Winning Trades: {wins}</li>
<li>Losing Trades: {losses}</li>
</ul>

<h2>Equity Curve</h2>
<canvas id="equityChart" height="150"></canvas>
<script>
new Chart(document.getElementById("equityChart"), {{
    type: 'line',
    data: {{
        labels: [...Array({len(equity_curve)}).keys()],
        datasets: [{{
            label: 'Equity',
            data: {equity_json},
            borderColor: 'blue',
            fill: false,
            tension: 0.1
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: true }} }},
        scales: {{
            x: {{ title: {{ display: true, text: 'Trade/Time Index' }} }},
            y: {{ title: {{ display: true, text: 'Equity ($)' }} }}
        }}
    }}
}});
</script>

<h2>Trade Log</h2>
<table>
<tr>
<th>Entry Date</th><th>Exit Date</th><th>Direction</th><th>Shares</th>
<th>Entry Price</th><th>Exit Price</th><th>PnL ($)</th><th>Equity After Close</th>
</tr>
{trade_rows}
</table>

</body>
</html>
"""

    filename = f"{symbol}_Backtest_Report.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report generated: {filename}")

# =========================================================
# Main
# =========================================================
def run():
    symbol = "BITX"
    df = prepare_data(symbol, months=3)  # <-- 最近120个月(10年)的日线生成30分钟CSV

    data = bt.feeds.PandasData(dataname=df)

    cerebro = bt.Cerebro()
    cerebro.adddata(data)

    params = dict(
        fast_period=9,
        slow_period=21,
        initial_shares=100,
        stop_loss_cash=300,
        take_profit_cash=300,
        recovery_mult=2,
    )

    cerebro.addstrategy(EMAStrategy, **params)
    cerebro.broker.setcash(100000)
    cerebro.broker.setcommission(commission=0.001)

    strat = cerebro.run()[0]

    pd.DataFrame(strat.trade_log).to_csv(f"{symbol}_trades.csv", index=False)
    generate_html(symbol, params, strat.equity_curve, strat.trade_log)

    print("Backtest finished")

if __name__ == "__main__":
    run()
