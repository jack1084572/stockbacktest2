import backtrader as bt
import pandas as pd

class TestStrategy(bt.Strategy):
    def __init__(self):
        self.trade_log = []

    def next(self):
        if not self.position:
            if self.data.close[0] > self.data.open[0]:  # simple long
                self.buy(size=100)
                self.entry_date = self.data.datetime.date(0)
        else:
            if self.data.close[0] < self.data.open[0]:  # simple exit
                self.sell(size=100)
                exit_date = self.data.datetime.date(0)
                self.trade_log.append({
                    "Entry Date": self.entry_date,
                    "Exit Date": exit_date,
                    "Entry Price": self.position.price,
                    "Exit Price": self.data.close[0],
                    "Size": self.position.size,
                    "PnL": (self.data.close[0] - self.position.price) * self.position.size
                })

# ---------------- Load CSV ----------------
data = bt.feeds.GenericCSVData(
    dataname='backtest_data.csv',
    dtformat='%Y-%m-%d',
    timeframe=bt.TimeFrame.Days,
    openinterest=-1,
    nullvalue=0.0,
    headers=True,
    open=5,    # Open 列索引
    high=3,    # High 列索引
    low=4,     # Low 列索引
    close=2,   # Close 列索引
    volume=6,  # Volume 列索引
    adjclose=1
)

# ---------------- Cerebro ----------------
cerebro = bt.Cerebro()
cerebro.addstrategy(TestStrategy)
cerebro.adddata(data)
cerebro.broker.setcash(100000)
cerebro.broker.setcommission(commission=0.001)

print("Starting Portfolio Value:", cerebro.broker.getvalue())
results = cerebro.run()
strategy_instance = results[0]
print("Final Portfolio Value:", cerebro.broker.getvalue())

# ---------------- Save Trades ----------------
trades_df = pd.DataFrame(strategy_instance.trade_log)
trades_df.to_csv('trades_out.csv', index=False)
print("Trades saved to trades_out.csv")
