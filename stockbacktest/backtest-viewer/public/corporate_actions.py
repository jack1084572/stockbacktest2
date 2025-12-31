import requests
import pandas as pd
import time
from datetime import datetime

# ============================
# 配置
# ============================
symbols = ["AAPL", "GAP", "SOXL"]  # 修改成你要的列表
sleep_seconds = 1.5

splits_file = "latest_splits.csv"
dividends_file = "latest_dividends.csv"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
}


# ============================
# 获取 Yahoo Finance JSON
# ============================
def fetch_yahoo_events(symbol):
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        "?range=10y&interval=1d&events=div%2Csplits"
    )

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"{symbol} fetch error: {e}")
        return None


# ============================
# 解析最近一次拆股
# ============================
def parse_latest_split(symbol, data):
    try:
        events = data["chart"]["result"][0].get("events", {})
        splits = events.get("splits", {})
        if not splits:
            return None

        # Yahoo splits 字典 key 为 timestamp
        # 取最近一个 timestamp
        latest_ts = max(splits.keys())
        latest = splits[latest_ts]

        ex_date = datetime.fromtimestamp(int(latest_ts)).strftime("%Y-%m-%d")
        num = latest.get("numerator")
        den = latest.get("denominator")

        ratio = f"{num}:{den}" if num is not None and den is not None else ""

        return {
            "symbol": symbol,
            "ex_date": ex_date,
            "ratio": ratio
        }
    except Exception:
        return None


# ============================
# 解析最近一次分红
# ============================
def parse_latest_dividend(symbol, data):
    try:
        events = data["chart"]["result"][0].get("events", {})
        divs = events.get("dividends", {})
        if not divs:
            return None

        latest_ts = max(divs.keys())
        latest = divs[latest_ts]

        ex_date = datetime.fromtimestamp(int(latest_ts)).strftime("%Y-%m-%d")
        amount = latest.get("amount")

        return {
            "symbol": symbol,
            "ex_date": ex_date,
            "amount": amount
        }
    except Exception:
        return None


# ============================
# 主程序
# ============================
def main():
    splits_out = []
    divs_out = []

    for sym in symbols:
        data = fetch_yahoo_events(sym)
        if data:
            split = parse_latest_split(sym, data)
            if split:
                splits_out.append(split)

            dividend = parse_latest_dividend(sym, data)
            if dividend:
                divs_out.append(dividend)

        time.sleep(sleep_seconds)

    if splits_out:
        pd.DataFrame(splits_out).to_csv(splits_file, index=False)
        print(f"Generated: {splits_file}")
    else:
        print("No split data found.")

    if divs_out:
        pd.DataFrame(divs_out).to_csv(dividends_file, index=False)
        print(f"Generated: {dividends_file}")
    else:
        print("No dividend data found.")


if __name__ == "__main__":
    main()
