# =========================================================
# BACKTRADER + YAHOO FINANCE + HTML REPORT (改进版)
# =========================================================

import os
import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# ================================
# 常量配置
# ================================
SYMBOL = "BITX"
TRADE_CSV = f"{SYMBOL}_trades.csv"
MODEL_FILE = "recovery_model.pkl"
SCALER_FILE = "recovery_scaler.pkl"

# 回测参数
BACKTEST_PARAMS = {
    "period": 20,
    "initial_shares": 100,
    "stop_loss_cash": 300,
    "take_profit_cash": 300,
    "recovery_mult": 2
}

# ================================
# ML 模型
# ================================
def ensure_model():
    """首次运行生成默认模型，避免 CSV 不存在报错"""
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        print("Model not found, creating default model...")
        default_model = RandomForestClassifier(n_estimators=10, random_state=42)
        default_model.fit([[0, 100, 100, 100]], [1])
        default_scaler = StandardScaler()
        default_scaler.fit([[0, 100, 100, 100]])
        joblib.dump(default_model, MODEL_FILE)
        joblib.dump(default_scaler, SCALER_FILE)
        print("Default model created.")

def predict_recovery(features):
    """预测回撤加仓成功概率"""
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        return 0.5
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    features_scaled = scaler.transform([features])
    prob = model.predict_proba(features_scaled)[0][1]
    return prob

# ================================
# 策略
# ================================
class LinearRegressionStrategy(bt.Strategy):
    params = dict(
        period=BACKTEST_PARAMS["period"],
        initial_shares=BACKTEST_PARAMS["initial_shares"],
        stop_loss_cash=BACKTEST_PARAMS["stop_loss_cash"],
        take_profit_cash=BACKTEST_PARAMS["take_profit_cash"],
        recovery_mult=BACKTEST_PARAMS["recovery_mult"],
        daily_data=None
    )

    def __init__(self):
        ensure_model()

        self.in_recovery = False
        self.recovery_shares = self.p.initial_shares
        self.last_trade_direction = None
        self.trade_log = []
        self.equity_curve = []
        self.equity_curve_dates = []
        self._prev_pos = 0
        self._entry = None

        if self.p.daily_data is None or len(self.p.daily_data) < self.p.period:
            raise ValueError("Daily data required and longer than period")

        self.daily_close = self.p.daily_data["close"].values

    def linear_regression_slope_daily(self):
        y = self.daily_close[-self.p.period:]
        x = np.arange(len(y))
        A = np.vstack([x, np.ones(len(x))]).T
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return slope

    def check_capital(self, shares):
        price = self.data.close[0]
        return abs(price * shares) <= self.broker.getvalue() * 0.9

    def next(self):
        # 每根 K 线记录权益曲线
        self.equity_curve.append(self.broker.getvalue())
        self.equity_curve_dates.append(self.data.datetime.datetime().strftime("%Y-%m-%d %H:%M"))

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
                # 回撤加仓逻辑
                if self.last_trade_direction == "LONG":
                    self.last_trade_direction = "SHORT"
                    self.sell(size=shares)
                else:
                    self.last_trade_direction = "LONG"
                    self.buy(size=shares)
        else:
            # 止盈止损
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
        if order.status != order.Completed:
            return

        new_pos = self.position.size
        prev_pos = self._prev_pos
        price = order.executed.price
        dt = bt.num2date(order.executed.dt).strftime("%Y-%m-%d %H:%M")

        if prev_pos == 0 and new_pos != 0:
            # 开仓
            self._entry = {
                "Entry Date": dt,
                "Direction": "LONG" if new_pos > 0 else "SHORT",
                "Shares": abs(new_pos),
                "Entry Price": price
            }
        elif prev_pos != 0 and new_pos == 0 and self._entry:
            # 平仓
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
                "PnL ($)": round(pnl, 2),
                "Equity After Close": round(self.broker.getvalue(), 2)
            })
            self._entry = None

        self._prev_pos = new_pos

# ================================
# 数据下载与读取
# ================================
def download_and_save_data(symbol, period, interval, filename):
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df["openinterest"] = 0
    df.to_csv(filename)
    print(f"Saved {interval} data: {filename}, rows: {len(df)}")
    return df

def load_csv_data(filename):
    df = pd.read_csv(filename, index_col=0, parse_dates=True)
    return df[["open", "high", "low", "close", "volume", "openinterest"]]

# ================================
# HTML 报告
# ================================
def generate_html_report(symbol, trades, equity_curve, equity_dates):
    total_pnl = sum(t["PnL ($)"] for t in trades)
    win_trades = sum(1 for t in trades if t["PnL ($)"] > 0)
    lose_trades = sum(1 for t in trades if t["PnL ($)"] <= 0)
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
<pre>{BACKTEST_PARAMS}</pre>

<h2>Statistics</h2>
<ul>
<li>Total Trades: {len(trades)}</li>
<li>Winning Trades: {win_trades}</li>
<li>Losing Trades: {lose_trades}</li>
<li>Win Rate: {win_rate:.2f}%</li>
<li>Total PnL: ${round(total_pnl,2)}</li>
<li>Final Equity: ${round(equity_curve[-1],2)}</li>
</ul>

<h2>Equity Curve</h2>
<canvas id="equityChart" width="800" height="400"></canvas>
<script>
var ctx = document.getElementById('equityChart').getContext('2d');
var equityChart = new Chart(ctx, {{
    type: 'line',
    data: {{
        labels: {equity_dates},
        datasets: [{{
            label: 'Equity Curve',
            data: {equity_curve},
            borderColor: 'rgb(75, 192, 192)',
            fill: false,
            tension: 0.1
        }}]
    }},
    options: {{
        responsive: true,
        scales: {{
            x: {{
                display: true,
                title: {{
                    display: true,
                    text: 'Date'
                }}
            }},
            y: {{
                display: true,
                title: {{
                    display: true,
                    text: 'Equity ($)'
                }}
            }}
        }}
    }}
}});
</script>

<h2>Trade Log</h2>
<table>
<tr>{''.join(f"<th>{k}</th>" for k in trades[0].keys())}</tr>
{''.join('<tr>' + ''.join(f"<td>{v}</td>" for v in t.values()) + '</tr>' for t in trades)}
</table>

</body>
</html>
"""
    with open(f"{symbol}.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report generated: {symbol}.html")

# ================================
# 主函数
# ================================
def main():
    csv_30m = f"{SYMBOL}_30m.csv"
    csv_1d = f"{SYMBOL}_1d.csv"

    df_30m = download_and_save_data(SYMBOL, "60d", "30m", csv_30m)
    df_1d = download_and_save_data(SYMBOL, "max", "1d", csv_1d)

    df_30m = load_csv_data(csv_30m)
    df_1d = load_csv_data(csv_1d)

    data = bt.feeds.PandasData(dataname=df_30m, timeframe=bt.TimeFrame.Minutes, compression=30)

    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(
        LinearRegressionStrategy,
        daily_data=df_1d
    )
    cerebro.broker.setcash(100000)
    cerebro.broker.setcommission(commission=0.001)

    strat = cerebro.run()[0]

    trades_df = pd.DataFrame(strat.trade_log)
    trades_df.to_csv(TRADE_CSV, index=False)
    print(f"Saved trades: {TRADE_CSV}, trades: {len(trades_df)}")

    if strat.trade_log:
        generate_html_report(SYMBOL, strat.trade_log, strat.equity_curve, strat.equity_curve_dates)

if __name__ == "__main__":
    main()
