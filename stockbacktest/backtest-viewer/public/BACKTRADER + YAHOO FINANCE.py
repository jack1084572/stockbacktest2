# =========================================================
# Imports
# =========================================================
import matplotlib
matplotlib.use("Agg")

import backtrader as bt
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Strategy
# =========================================================
class SmaCrossStrategy(bt.Strategy):
    params = dict(
        fast=10,
        slow=30,

        initial_shares=None,
        max_capital_usage_pct=None,

        stop_loss_cash=None,
        take_profit_cash=None,

        recovery_mult=None
    )

    def __init__(self):
        self.fast_sma = bt.indicators.SMA(self.data.close, period=self.p.fast)
        self.slow_sma = bt.indicators.SMA(self.data.close, period=self.p.slow)

        self.in_recovery = False
        self.recovery_shares = self.p.initial_shares
        self.last_trade_direction = None

        self.trade_log = []
        self.equity_curve = []

        self._prev_pos = 0
        self._entry = None

    # ------------------------------
    # 资金检查
    # ------------------------------
    def check_capital(self, shares):
        price = self.data.close[0]
        return abs(price * shares) <= self.broker.getvalue() * self.p.max_capital_usage_pct

    # ------------------------------
    # 主逻辑
    # ------------------------------
    def next(self):
        self.equity_curve.append(self.broker.getvalue())

        # ===== 开仓 =====
        if not self.position:
            if not self.in_recovery:
                shares = self.p.initial_shares
            else:
                shares = self.recovery_shares

            if not self.check_capital(shares):
                return

            if not self.in_recovery:
                if self.fast_sma[0] > self.slow_sma[0]:
                    self.last_trade_direction = "LONG"
                    self.buy(size=shares)
                elif self.fast_sma[0] < self.slow_sma[0]:
                    self.last_trade_direction = "SHORT"
                    self.sell(size=shares)
            else:
                if self.last_trade_direction == "LONG":
                    self.last_trade_direction = "SHORT"
                    self.sell(size=shares)
                else:
                    self.last_trade_direction = "LONG"
                    self.buy(size=shares)

        # ===== 固定金额止盈止损（Recovery 动态缩放）=====
        else:
            entry_price = self.position.price
            price = self.data.close[0]
            shares = abs(self.position.size)

            pnl = (
                (price - entry_price) * shares
                if self.position.size > 0
                else (entry_price - price) * shares
            )

            scale = shares / self.p.initial_shares

            stop_loss_cash = self.p.stop_loss_cash * scale
            take_profit_cash = self.p.take_profit_cash * scale

            if pnl <= -stop_loss_cash:
                self.close()
            elif pnl >= take_profit_cash:
                self.close()

    # ------------------------------
    # 成交回调（CSV 唯一来源）
    # ------------------------------
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            new_pos = self.position.size
            prev_pos = self._prev_pos

            price = order.executed.price
            dt = bt.num2date(order.executed.dt).strftime("%Y-%m-%d %H:%M")

            # ===== 开仓 =====
            if prev_pos == 0 and new_pos != 0:
                self._entry = {
                    "Entry Date": dt,
                    "Direction": "LONG" if new_pos > 0 else "SHORT",
                    "Shares": abs(new_pos),
                    "Entry Price": price
                }

            # ===== 平仓 =====
            elif prev_pos != 0 and new_pos == 0 and self._entry:
                qty = self._entry["Shares"]

                if self._entry["Direction"] == "LONG":
                    pnl = (price - self._entry["Entry Price"]) * qty
                else:
                    pnl = (self._entry["Entry Price"] - price) * qty

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
                    "PnL ($)": pnl
                })

                self._entry = None

            self._prev_pos = new_pos


# =========================================================
# Data
# =========================================================
def get_minute_data(symbol, period, interval):
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

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
    return df[["open", "high", "low", "close", "volume", "openinterest"]]


# =========================================================
# Main
# =========================================================
def main():
    # ====== 运行时参数（只在这里设置） ======
    stock_symbol = "SOXL"

    initial_shares = 100
    max_capital_usage_pct = 0.9

    initial_capital = 100000

    stop_loss_cash = initial_shares*4      # 单笔基础止损（美元）
    take_profit_cash = stop_loss_cash    # 单笔基础止盈（美元）

    recovery_mult = 2

    period = "60d"
    interval = "30m"

    # ====== 数据 ======
    data = bt.feeds.PandasData(
        dataname=get_minute_data(stock_symbol, period, interval),
        timeframe=bt.TimeFrame.Minutes,
        compression=30
    )

    # ====== 回测引擎 ======
    cerebro = bt.Cerebro()
    cerebro.adddata(data)

    cerebro.addstrategy(
        SmaCrossStrategy,
        initial_shares=initial_shares,
        max_capital_usage_pct=max_capital_usage_pct,
        stop_loss_cash=stop_loss_cash,
        take_profit_cash=take_profit_cash,
        recovery_mult=recovery_mult
    )

    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=0.001)

    strat = cerebro.run()[0]

    # ====== 输出 ======
    pd.DataFrame(strat.trade_log).to_csv("trades.csv", index=False)

    plt.plot(strat.equity_curve)
    plt.title("Equity Curve")
    plt.savefig("equity_curve.png")


if __name__ == "__main__":
    main()
