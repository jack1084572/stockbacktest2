# =========================================================
# Imports
# =========================================================
import matplotlib
matplotlib.use("Agg")

import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np

# =========================================================
# Strategy: EMA Crossover (金叉多/死叉空)
# =========================================================
# =========================================================
# Strategy: EMA Compare (快线>慢线做多, 快线<慢线做空)
# =========================================================
class EMACrossoverStrategy(bt.Strategy):
    params = dict(
        fast_period=9,
        slow_period=21,
        initial_shares=100,
        max_capital_usage_pct=0.9,
        stop_loss_cash=100,
        take_profit_cash=100,
        recovery_mult=2
    )

    def __init__(self):
        self.in_recovery = False
        self.recovery_shares = self.p.initial_shares

        self.trade_log = []
        self.equity_curve = []

        self._prev_pos = 0
        self._entry = None

        self.ema_fast = bt.ind.EMA(self.data.close, period=self.p.fast_period)
        self.ema_slow = bt.ind.EMA(self.data.close, period=self.p.slow_period)

    def check_capital(self, shares):
        price = self.data.close[0]
        return abs(price * shares) <= self.broker.getvalue() * self.p.max_capital_usage_pct

    def next(self):
        self.equity_curve.append(self.broker.getvalue())

        fast = self.ema_fast[0]
        slow = self.ema_slow[0]

        # 当前快线 > 慢线 做多, 否则做空
        if not self.position:
            shares = self.recovery_shares if self.in_recovery else self.p.initial_shares
            if not self.check_capital(shares):
                return

            if fast > slow:
                self.buy(size=shares)
            else:
                self.sell(size=shares)
        else:
            entry_price = self.position.price
            price = self.data.close[0]
            shares = abs(self.position.size)
            pnl = (price - entry_price) * shares if self.position.size > 0 else (entry_price - price) * shares
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
                    "Entry Price": price
                }

            elif prev_pos != 0 and new_pos == 0 and self._entry:
                qty = self._entry["Shares"]
                pnl = ((price - self._entry["Entry Price"]) * qty
                       if self._entry["Direction"] == "LONG"
                       else (self._entry["Entry Price"] - price) * qty)

                # 回撤加仓逻辑
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
                    "PnL ($)": round(pnl, 2),
                    "Equity After Close": round(self.broker.getvalue(), 2)
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
    return df[["open", "high", "low", "close", "volume", "openinterest"]]

# =========================================================
# HTML Report (优化多空和盈亏颜色)
# =========================================================
def generate_html_report(symbol, params, equity_curve, trades):
    import json

    # 统计数据
    total_pnl = sum(t["PnL ($)"] for t in trades)
    win_trades = sum(1 for t in trades if t["PnL ($)"] > 0)
    loss_trades = len(trades) - win_trades
    win_rate = (win_trades / len(trades) * 100) if trades else 0

    # 策略参数
    params_html = "<ul>" + "".join(f"<li>{k}: {v}</li>" for k, v in params.items()) + "</ul>"

    # Trade Log 表格
    trade_log_html = f"""
<style>
table.trade-log {{
    border-collapse: collapse;
    width: 100%;
    font-family: Arial, sans-serif;
}}
table.trade-log th, table.trade-log td {{
    border: 1px solid #ccc;
    padding: 6px;
    text-align: center;
    font-size: 13px;
}}
table.trade-log th {{
    background-color: #f2f2f2;
    font-weight: bold;
}}
table.trade-log tr:nth-child(even) {{
    background-color: #fafafa;
}}
table.trade-log tr.win td {{
    background-color: #d4edda;  /* 盈利绿色 */
}}
table.trade-log tr.loss td {{
    background-color: #f8d7da;  /* 亏损红色 */
}}
table.trade-log tr.short td {{
    color: red;  /* 空单字体红色 */
    font-weight: bold;
}}
table.trade-log tr.long td {{
    color: green;  /* 多单字体绿色 */
    font-weight: bold;
}}
</style>
<table class="trade-log">
<tr>{''.join(f"<th>{k}</th>" for k in trades[0].keys())}</tr>
{''.join(
    '<tr class="{} {}">'.format(
        'win' if t["PnL ($)"] > 0 else 'loss',
        'long' if t["Direction"]=="LONG" else 'short'
    ) +
    ''.join(f"<td>{v}</td>" for v in t.values()) +
    '</tr>' for t in trades
)}
</table>
"""

    # Equity 曲线 JSON
    equity_json = json.dumps(equity_curve)

    # HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{symbol} Backtest Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body style="font-family: Arial, sans-serif; margin: 20px;">

<h1>{symbol} Backtest Report</h1>

<h2>Strategy Parameters</h2>
{params_html}

<h2>Statistics</h2>
<ul>
<li>Total Trades: {len(trades)}</li>
<li>Win Rate: {win_rate:.2f}%</li>
<li>Total PnL: {round(total_pnl,2)}</li>
<li>Winning Trades: {win_trades}</li>
<li>Losing Trades: {loss_trades}</li>
</ul>

<h2>Equity Curve</h2>
<canvas id="equityChart" width="1000" height="400"></canvas>
<script>
const equityData = {equity_json};
const ctx = document.getElementById('equityChart').getContext('2d');
new Chart(ctx, {{
    type: 'line',
    data: {{
        labels: equityData.map((_, i) => i+1),
        datasets: [{{
            label: 'Equity',
            data: equityData,
            borderColor: 'blue',
            backgroundColor: 'rgba(0,0,255,0.1)',
            fill: true,
            tension: 0.1
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{
            legend: {{ display: true }},
        }},
        scales: {{
            x: {{ title: {{ display: true, text: 'Trade/Time Index' }} }},
            y: {{ title: {{ display: true, text: 'Equity ($)' }} }}
        }}
    }}
}});
</script>

<h2>Trade Log</h2>
{trade_log_html}

</body>
</html>
"""

    with open(f"{symbol}.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report generated: {symbol}.html")

# =========================================================
# Main
# =========================================================
def main():
    stock_symbol = "SOXL"

    csv_30m = f"{stock_symbol}_30m.csv"
    df_30m = download_and_save_data(stock_symbol, "60d", "30m", csv_30m)
    df_30m = load_csv_data(csv_30m)

    data = bt.feeds.PandasData(dataname=df_30m, timeframe=bt.TimeFrame.Minutes, compression=30)

    cerebro = bt.Cerebro()
    cerebro.adddata(data)

    # 策略参数
    strat_params = dict(
        fast_period=9,
        slow_period=21,
        initial_shares=100,
        stop_loss_cash=300,
        take_profit_cash=300,
        recovery_mult=2
    )

    cerebro.addstrategy(EMACrossoverStrategy, **strat_params)

    cerebro.broker.setcash(100000)
    cerebro.broker.setcommission(commission=0.001)

    strat = cerebro.run()[0]

    trades_df = pd.DataFrame(strat.trade_log)
    trades_df.to_csv(f"{stock_symbol}_trades.csv", index=False)
    print(f"Saved trades: {stock_symbol}_trades.csv, trades: {len(trades_df)}")

    generate_html_report(stock_symbol, strat_params, strat.equity_curve, strat.trade_log)

if __name__ == "__main__":
    main()
