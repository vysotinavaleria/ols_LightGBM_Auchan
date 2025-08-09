# -*- coding: utf-8 -*-
"""
Прогноз средневзвешенной задержки оплаты (календарные дни) для августа/сентября 2025.
Чтение данных с листа Sheet2, LightGBM (с фолбэком), расширенные фичи и прогресс-бары.

Запуск:
    python main.py "c:/Users/bayon/Desktop/ols/for model (2).xlsx"

Если путь не передан — возьмём файл "for model (2).xlsx" из папки со скриптом.
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -------- tqdm (прогресс-бар) --------
try:
    from tqdm import tqdm
except Exception:
    # запасной вариант, если tqdm не установлен
    def tqdm(iterable=None, total=None, desc=None):
        return iterable if iterable is not None else []

# ---- LightGBM или безопасный фолбэк ----
use_lgbm = True
try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
except Exception:
    use_lgbm = False
    from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------- настройки чтения -------------------
SHEET_TO_READ = "Sheet2"

RECV_CANDS = ["journal entry date", "дата получения", "journal_entry_date", "entry date", "entry_date"]
PAY_CANDS  = ["clearing date", "дата оплаты", "payment date", "payment_date"]
AMT_CANDS  = ["sum w/o vat", "amount", "сумма", "стоимость", "total", "amount_uah", "amount_gbp"]

# ------------------- утилиты -------------------
def load_excel_robust(p: Path) -> pd.DataFrame:
    """Читаем именно Sheet2 (или индекс 1), поднимаем «кривую» шапку при необходимости."""
    print("📥 Загрузка Excel...")
    try:
        df0 = pd.read_excel(p, sheet_name=SHEET_TO_READ)
    except Exception:
        df0 = pd.read_excel(p, sheet_name=1)

    cols = [str(c).strip() for c in df0.columns]
    if all(str(c).startswith("Unnamed:") for c in cols):
        try:
            df = pd.read_excel(p, sheet_name=SHEET_TO_READ, header=1)
        except Exception:
            df = pd.read_excel(p, sheet_name=1, header=1)
        if df.shape[0] > 0 and any(str(v).lower() in {"journal entry date", "clearing date", "sum w/o vat"} for v in df.iloc[0].values):
            new_header = df.iloc[0]
            df = df[1:].copy()
            df.columns = new_header.values
        df = df.reset_index(drop=True)
    else:
        df = df0

    print(f"✅ Загрузка готова. Строк: {len(df)}, колонок: {len(df.columns)}")
    return df

def find_col(candidates, cols):
    cl = [str(c).lower() for c in cols]
    for cand in candidates:
        if cand in cl:
            return cols[cl.index(cand)]
    return None

def parse_date_series(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, format="%d.%m.%Y", errors="coerce")
    if out.isna().all():
        out = pd.to_datetime(s, errors="coerce")
    return out

# ------------------- построение фич (календарные дни) -------------------
def build_features(df_raw: pd.DataFrame, recv_col: str, pay_col: str, amt_col: str):
    print("🧪 Подготовка фич...")
    df = df_raw.copy()

    df[recv_col] = parse_date_series(df[recv_col])
    df[pay_col]  = parse_date_series(df[pay_col])
    df["amount"] = pd.to_numeric(df[amt_col], errors="coerce")

    df["days_to_pay"] = (df[pay_col] - df[recv_col]).dt.days

    df = df.dropna(subset=[recv_col, pay_col, "amount", "days_to_pay"]).copy()
    df = df[df["days_to_pay"] >= 0]

    rdt = df[recv_col]
    df["year"] = rdt.dt.year
    df["month"] = rdt.dt.month
    df["day"] = rdt.dt.day
    df["dayofweek"] = rdt.dt.dayofweek
    df["quarter"] = rdt.dt.quarter
    df["day_of_year"] = rdt.dt.dayofyear
    df["week_of_year"] = rdt.dt.isocalendar().week.astype(int)

    def season(m):
        if m in (12, 1, 2):
            return 1
        if m in (3, 4, 5):
            return 2
        if m in (6, 7, 8):
            return 3
        return 4
    df["season"] = df["month"].apply(season).astype(int)

    eom = (rdt + pd.offsets.MonthEnd(0))
    eoq = (rdt + pd.offsets.QuarterEnd(0))
    som = (rdt - pd.offsets.MonthBegin(0))
    soq = (rdt - pd.offsets.QuarterBegin(0))

    df["is_month_end"] = rdt.dt.is_month_end.astype(int)
    df["is_quarter_end"] = rdt.dt.is_quarter_end.astype(int)
    df["days_to_month_end"] = (eom - rdt).dt.days
    df["days_from_month_start"] = (rdt - som).dt.days
    df["days_to_quarter_end"] = (eoq - rdt).dt.days
    df["days_from_quarter_start"] = (rdt - soq).dt.days

    df["is_negative_amount"] = (df["amount"] < 0).astype(int)
    amt_for_log = df["amount"].clip(lower=0)
    df["log_amount"] = np.log1p(amt_for_log)
    df["amount_sq"] = df["amount"] ** 2

    q = df["amount"].quantile([0.2, 0.4, 0.6, 0.8])
    df["amt_bin"] = pd.cut(df["amount"],
                           bins=[-np.inf, q.iloc[0], q.iloc[1], q.iloc[2], q.iloc[3], np.inf],
                           labels=[0, 1, 2, 3, 4]).astype(int)

    df["log_amount_x_month"] = df["log_amount"] * df["month"]
    df["log_amount_x_quarter"] = df["log_amount"] * df["quarter"]
    df["month_x_dow"] = df["month"] * df["dayofweek"]

    df["y"] = df["days_to_pay"].clip(0, df["days_to_pay"].quantile(0.99))

    feature_cols = [
        "amount", "log_amount", "amount_sq", "is_negative_amount", "amt_bin",
        "year", "month", "day", "quarter", "dayofweek", "day_of_year", "week_of_year", "season",
        "is_month_end", "is_quarter_end",
        "days_to_month_end", "days_from_month_start",
        "days_to_quarter_end", "days_from_quarter_start",
        "log_amount_x_month", "log_amount_x_quarter", "month_x_dow",
    ]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df["y"].astype(float).copy()

    print(f"✅ Фичи готовы. Строк: {len(df)}, признаков: {len(feature_cols)}")
    return df, X, y, feature_cols

def time_split(df, X, y, recv_col, frac=0.8):
    print("✂️  Сплит по времени (80/20)...")
    sorted_idx = df.sort_values(by=recv_col).index
    split_idx = int(len(sorted_idx) * frac)
    train_idx = sorted_idx[:split_idx]
    test_idx  = sorted_idx[split_idx:]
    print(f"✅ Трейн: {len(train_idx)}  Тест: {len(test_idx)}")
    return X.loc[train_idx], y.loc[train_idx], X.loc[test_idx], y.loc[test_idx], train_idx

# ------------------- обучение -------------------
def train_model(X_train, y_train, X_test, y_test):
    print("🚀 Обучение модели...")
    if use_lgbm:
        model = LGBMRegressor(
            n_estimators=1400,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=39,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.2,
            random_state=42
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  eval_metric="l1",
                  verbose=False)
    else:
        model = HistGradientBoostingRegressor(
            max_depth=None,
            learning_rate=0.05,
            max_iter=1000,
            l2_regularization=0.0,
            random_state=42
        )
        model.fit(X_train, y_train)
    print("✅ Обучение завершено")
    return model

# ------------------- утилиты прогнозов -------------------
def month_days(year: int, month: int):
    d = date(year, month, 1)
    res = []
    while d.month == month:
        res.append(pd.Timestamp(d))
        d += timedelta(days=1)
    return res

def build_feature_row(ts: pd.Timestamp, amount_value: float, amount_bins, feature_cols: list):
    quarter = int((ts.month - 1)//3 + 1)
    day_of_year = ts.timetuple().tm_yday
    week_of_year = pd.Timestamp(ts).isocalendar().week
    season = 1 if ts.month in (12,1,2) else 2 if ts.month in (3,4,5) else 3 if ts.month in (6,7,8) else 4

    is_month_end = int(pd.Timestamp(ts).is_month_end)
    is_quarter_end = int(pd.Timestamp(ts).is_quarter_end)

    eom = ts + pd.offsets.MonthEnd(0)
    som = ts - pd.offsets.MonthBegin(0)
    eoq = ts + pd.offsets.QuarterEnd(0)
    soq = ts - pd.offsets.QuarterBegin(0)

    is_neg = int(amount_value < 0)
    amt_for_log = max(amount_value, 0.0)
    log_amount = np.log1p(amt_for_log)
    amount_sq = float(amount_value) ** 2

    b0, b1, b2, b3 = amount_bins
    if amount_value <= b0:
        amt_bin = 0
    elif amount_value <= b1:
        amt_bin = 1
    elif amount_value <= b2:
        amt_bin = 2
    elif amount_value <= b3:
        amt_bin = 3
    else:
        amt_bin = 4

    feats = {
        "amount": float(amount_value),
        "log_amount": log_amount,
        "amount_sq": amount_sq,
        "is_negative_amount": is_neg,
        "amt_bin": int(amt_bin),

        "year": ts.year,
        "month": ts.month,
        "day": ts.day,
        "quarter": quarter,
        "dayofweek": ts.dayofweek,
        "day_of_year": int(day_of_year),
        "week_of_year": int(week_of_year),
        "season": int(season),

        "is_month_end": is_month_end,
        "is_quarter_end": is_quarter_end,
        "days_to_month_end": int((eom - ts).days),
        "days_from_month_start": int((ts - som).days),
        "days_to_quarter_end": int((eoq - ts).days),
        "days_from_quarter_start": int((ts - soq).days),

        "log_amount_x_month": log_amount * ts.month,
        "log_amount_x_quarter": log_amount * quarter,
        "month_x_dow": ts.month * ts.dayofweek,
    }
    x = pd.DataFrame([feats]).reindex(columns=feature_cols, fill_value=0.0)
    return x

def monthly_weighted_avg(model, year, month, train_amounts: np.ndarray, amount_bins, feature_cols, max_amounts: int = 4000):
    """
    Средневзвешенные дни за месяц: веса = сумма сделки.
    Распределение сумм берём из тренировки (train_amounts).
    Показываем прогресс-бар по суммам.
    """
    days = month_days(year, month)
    if not days:
        return float("nan")

    # очистим и, при необходимости, подсэмплим суммы (для скорости)
    arr = pd.to_numeric(pd.Series(train_amounts), errors="coerce").dropna().values
    if arr.size == 0:
        return float("nan")

    if arr.size > max_amounts:
        abs_amt = np.abs(arr)
        probs = abs_amt / abs_amt.sum()
        idx = np.random.choice(np.arange(arr.size), size=max_amounts, replace=False, p=probs)
        arr = arr[idx]

    weighted_sum = 0.0
    total_weight = 0.0

    for amt in tqdm(arr, desc=f"📦 Прогноз для {year}-{month:02d} (взвешивание по суммам)", total=len(arr)):
        preds_for_amt = []
        for ts in days:
            x = build_feature_row(ts, float(amt), amount_bins, feature_cols)
            preds_for_amt.append(float(model.predict(x)[0]))
        mean_for_amt = float(np.mean(preds_for_amt))
        weight = float(abs(amt))
        weighted_sum += mean_for_amt * weight
        total_weight += weight

    return (weighted_sum / total_weight) if total_weight > 0 else float("nan")

# ------------------- main -------------------
def main():
    # путь к файлу
    if len(sys.argv) < 2:
        p = Path(__file__).with_name("for model (2).xlsx")
        print(f'Путь не передан, пробую: "{p}"')
    else:
        p = Path(sys.argv[1])
    if not p.exists():
        print(f'❌ Файл не найден: "{p}"')
        sys.exit(1)

    # загрузка
    df_raw = load_excel_robust(p)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    recv_col = find_col(RECV_CANDS, df_raw.columns) or df_raw.columns[0]
    pay_col  = find_col(PAY_CANDS,  df_raw.columns) or df_raw.columns[1]
    amt_col  = find_col(AMT_CANDS,  df_raw.columns) or df_raw.columns[2]
    print(f"ℹ️  Используем колонки: recv='{recv_col}', pay='{pay_col}', amount='{amt_col}'")

    # фичи и сплит
    df, X, y, feature_cols = build_features(df_raw, recv_col, pay_col, amt_col)
    X_train, y_train, X_test, y_test, train_idx = time_split(df, X, y, recv_col, 0.8)

    # обучение
    model = train_model(X_train, y_train, X_test, y_test)

    # метрики
    print("📏 Оценка качества...")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    print(f"✅ MAE (тест):  {mae:.2f} дней")
    print(f"✅ RMSE (тест): {rmse:.2f} дней\n")

    # веса = распределение сумм в тренировке
    train_amounts = df.loc[train_idx, "amount"].values

    # пороги квантилей train для бинов суммы (для совместимости фич при predict)
    q = df.loc[train_idx, "amount"].quantile([0.2, 0.4, 0.6, 0.8])
    amount_bins = (float(q.iloc[0]), float(q.iloc[1]), float(q.iloc[2]), float(q.iloc[3]))

    # средневзвешенные дни
    wavg_aug = monthly_weighted_avg(model, 2025, 8, train_amounts, amount_bins, feature_cols, max_amounts=4000)
    wavg_sep = monthly_weighted_avg(model, 2025, 9, train_amounts, amount_bins, feature_cols, max_amounts=4000)

    print("\n📣 Средневзвешенная ожидаемая задержка (календарные дни до оплаты), вес = сумма сделки:")
    print(f"  Август 2025:   {wavg_aug:.2f} дней")
    print(f"  Сентябрь 2025: {wavg_sep:.2f} дней")

    print("\n✅ Расчёты завершены")

if __name__ == "__main__":
    main()
