# =========================================================
# train_recovery_model.py
# =========================================================

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

SYMBOL = "TQQQ"
TRADE_CSV = f"{SYMBOL}_trades.csv"
MODEL_FILE = "recovery_model.pkl"
SCALER_FILE = "recovery_scaler.pkl"
LOG_FILE = "recovery_decisions.csv"


def train_model():
    # CSV 不存在或为空 → 创建默认模型
    if not os.path.exists(TRADE_CSV) or os.path.getsize(TRADE_CSV) == 0:
        print("Trade CSV missing or empty, creating default model...")
        # 默认模型：随机森林，不训练，用于首次回测
        default_model = RandomForestClassifier(n_estimators=10, random_state=42)
        default_model.fit([[0, 100, 100, 100]], [1])  # 单条虚拟数据
        default_scaler = StandardScaler()
        default_scaler.fit([[0, 100, 100, 100]])
        joblib.dump(default_model, MODEL_FILE)
        joblib.dump(default_scaler, SCALER_FILE)
        print(f"Default model created: {MODEL_FILE}")
        return

    df = pd.read_csv(TRADE_CSV)
    if len(df) < 5:
        print("Not enough trades to train model, creating default model...")
        # 同上，生成默认模型
        default_model = RandomForestClassifier(n_estimators=10, random_state=42)
        default_model.fit([[0, 100, 100, 100]], [1])
        default_scaler = StandardScaler()
        default_scaler.fit([[0, 100, 100, 100]])
        joblib.dump(default_model, MODEL_FILE)
        joblib.dump(default_scaler, SCALER_FILE)
        return

    # Feature engineering
    df['ConsecutiveLoss'] = (df['PnL ($)'] < 0).astype(int)
    df['Target'] = (df['PnL ($)'] > 0).astype(int)

    X = df[['ConsecutiveLoss', 'Shares', 'Entry Price', 'Exit Price']].values
    y = df['Target'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"RandomForest model trained and saved: {MODEL_FILE}")


def predict_recovery(features):
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        return 0.5  # 模型不存在，返回中性概率

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    features_scaled = scaler.transform([features])
    prob = model.predict_proba(features_scaled)[0][1]
    return prob


def log_recovery_decision(decision_record):
    if not os.path.exists(LOG_FILE):
        pd.DataFrame([decision_record]).to_csv(LOG_FILE, index=False)
    else:
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([df, pd.DataFrame([decision_record])], ignore_index=True)
        df.to_csv(LOG_FILE, index=False)


if __name__ == "__main__":
    train_model()
