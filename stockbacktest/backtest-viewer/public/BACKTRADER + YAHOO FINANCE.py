# =========================================================
# Imports
# =========================================================
import pandas as pd
import backtrader as bt
import json
from dateutil.relativedelta import relativedelta
import pytz
from datetime import time

# =========================================================
# Strategy: EMA + Recovery + Reverse Add-on + NYSE Time Limit
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
        ny_open=time(9,30),
        ny_close=time(16,0),
    )

    def __init__(self):
        self.ema_fast = bt.ind.EMA(self.data.close, period=self.p.fast_period)
        self.ema_slow = bt.ind.EMA(self.data.close, period=self.p.slow_period)
        self.trade_log = []
        self.equity_curve = []
        self._entry = None
        self.in_recovery = False
        self.recovery_shares = self.p.initial_shares
        self.last_trade_direction = None
        self.ny_tz = pytz.timezone("US/Eastern")  # 纽约时区

    def check_capital(self, shares):
        price = self.data.close[0]
        return abs(price * shares) <= self.broker.getvalue() * self.p.max_capital_pct

    def is_nyse_open(self):
        dt = self.data.datetime.datetime(0)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC)
        dt = dt.astimezone(self.ny_tz)
        return self.p.ny_open <= dt.time() <= self.p.ny_close

    def next(self):
        # 每根K线都记录账户净值
        self.equity_curve.append(round(self.broker.getvalue(), 2))

        # 非交易时间直接跳过
        if not self.is_nyse_open():
            return

        fast = self.ema_fast[0]
        slow = self.ema_slow[0]
        price = self.data.close[0]

        # =========================
        # 持仓止损/止盈
        # =========================
        if self.position:
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

        # =========================
        # 反向加仓循环
        # =========================
        if self.in_recovery and not self.position and self.last_trade_direction:
            shares = self.recovery_shares
            if self.last_trade_direction == "LONG" and self.check_capital(shares):
                self.sell(size=shares)
            elif self.last_trade_direction == "SHORT" and self.check_capital(shares):
                self.buy(size=shares)

        # =========================
        # 正常 EMA 策略开仓
        # =========================
        if not self.position and not self.in_recovery:
            shares = self.p.initial_shares
            if not self.check_capital(shares):
                return
            if fast > slow:
                self.buy(size=shares)
            elif fast < slow:
                self.sell(size=shares)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status != order.Completed:
            return

        price = order.executed.price
        size = order.executed.size
        direction = "LONG" if size > 0 else "SHORT"

        dt = bt.num2date(order.executed.dt)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC)
        dt = dt.astimezone(self.ny_tz)
        dt_str = dt.strftime("%Y-%m-%d %H:%M")

        # =========================
        # 只在 NYSE 交易时段记录
        # =========================
        if not (self.p.ny_open <= dt.time() <= self.p.ny_close):
            return

        # 开仓/加仓记录
        if self.position.size != 0:
            if self._entry is None:
                self._entry = {
                    "Entry Date": dt_str,
                    "Direction": direction,
                    "Shares": abs(size),
                    "Entry Price": round(price, 2),
                }
            else:
                prev_qty = self._entry["Shares"]
                prev_price = self._entry["Entry Price"]
                new_qty = abs(size)
                weighted_price = (prev_price * prev_qty + price * new_qty) / (prev_qty + new_qty)
                self._entry["Shares"] += new_qty
                self._entry["Entry Price"] = round(weighted_price, 2)

        # 平仓记录
        if self.position.size == 0 and self._entry:
            qty = self._entry["Shares"]
            pnl = (
                (price - self._entry["Entry Price"]) * qty
                if self._entry["Direction"] == "LONG"
                else (self._entry["Entry Price"] - price) * qty
            )

            if pnl < 0:
                self.in_recovery = True
                self.recovery_shares = self._entry["Shares"] * self.p.recovery_mult
            else:
                self.in_recovery = False
                self.recovery_shares = self.p.initial_shares

            self.last_trade_direction = self._entry["Direction"]

            self.trade_log.append({
                "Entry Date": self._entry["Entry Date"],
                "Exit Date": dt_str,
                "Direction": self._entry["Direction"],
                "Shares": qty,
                "Entry Price": self._entry["Entry Price"],
                "Exit Price": round(price, 2),
                "PnL ($)": round(pnl, 2),
                "Equity After Close": round(self.broker.getvalue(), 2),
            })

            self._entry = None

# =========================================================
# CSV Loader
# =========================================================
def load_m30_csv(file_path, months=None):
    df = pd.read_csv(file_path, sep="\t")
    df['datetime'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
    # 假设 CSV 原始时间是 UTC
    df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    df = df.set_index('datetime')
    df = df.rename(columns={
        '<OPEN>': 'open',
        '<HIGH>': 'high',
        '<LOW>': 'low',
        '<CLOSE>': 'close',
        '<TICKVOL>': 'volume',
        '<VOL>': 'openinterest'
    })
    df = df[['open','high','low','close','volume','openinterest']]
    if months:
        start_date = df.index[-1] - relativedelta(months=months)
        df = df[df.index >= start_date]
    return df

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
# Run
# =========================================================
def run():
    symbol = "SOXL"
    file_path = "BITX_M30_202306271630_202601072230.csv"
    months_to_backtest = 6  # 最近 N 个月回测

    df = load_m30_csv(file_path, months=months_to_backtest)
    data = bt.feeds.PandasData(dataname=df)

    cerebro = bt.Cerebro()
    cerebro.adddata(data)

    params = dict(
        fast_period=9,
        slow_period=21,
        initial_shares=100,
        stop_loss_cash=500,
        take_profit_cash=500,
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
