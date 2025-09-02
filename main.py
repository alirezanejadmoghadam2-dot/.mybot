# -*- coding: utf-8 -*-
import time
from datetime import datetime, timezone, timedelta
import jdatetime
import ccxt
import numpy as np
import pandas as pd
import requests
import threading

# ==============================
# 1) INPUT PARAMETERS (عین Pine)
# ==============================
countbc = 3

# RSI & Length
length = 21
rsi_length = length
rsi_sell = 60.0
rsi_buy = 40.0

# MACD (TF اصلی)
macd_fast_length = 9
macd_slow_length = 26
macd_signal_length = 12
macd_threshold = 400.0

# ADX
adx_val = 20.0
adx_length = length
adx_smoothing = length

# Squeeze Momentum (LazyBear)
sqz_length = 20
sqz_mult = 2.0
kc_length = 20
kc_mult = 1.5
useTrueRange = True
sqzbuy = -700.0
sqzsell = 700.0

# ================
# MTF MACD (HTF)
# ================
useCurrentRes = False
resCustom = "1d"
mtf_buy_threshold = -700.0
mtf_sell_threshold = 700.0
fastLength_mtf = 12
slowLength_mtf = 26
signalLength_mtf = 9

# ==============================
# 2) EXCHANGE / DATA SETTINGS
# ==============================
symbol = "BTC/USDT"
tf = "15m"
limit_15m = 5000
poll_seconds = 300
LIVE = True

# --- تاخیر ارسال به کانال عمومی بر حسب ساعت
delay_public_hours = 2.5
delay_public_seconds = int(delay_public_hours * 3600)

# ناحیه زمانی محلی (UTC+5:12)
LOCAL_TZ = timezone(timedelta(hours=5, minutes=12))

# اتصال KuCoin
exchange = ccxt.kucoin({
    "timeout": 60000,
    "enableRateLimit": True,
})

# ==============================
# 3) HELPER FUNCTIONS
# ==============================
def rma(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0/period, adjust=False).mean()

def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = pd.Series(np.where(delta > 0, delta, 0.0), index=series.index)
    down = pd.Series(np.where(delta < 0, -delta, 0.0), index=series.index)
    rs = rma(up, period) / rma(down, period)
    return 100 - (100/(1+rs))

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def adx_plus_minus_di(high, low, close, length_adx: int, smoothing: int):
    up = high.diff()
    down = -low.diff()
    plusDM = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minusDM = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    tr_rma = rma(true_range(high, low, close), length_adx)
    plusDI = 100.0 * rma(plusDM, length_adx) / tr_rma
    minusDI = 100.0 * rma(minusDM, length_adx) / tr_rma
    dx = 100.0 * (plusDI - minusDI).abs() / (plusDI + minusDI)
    adx_val_series = rma(dx, smoothing)
    return plusDI, minusDI, adx_val_series

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length, min_periods=length).mean()

def macd_lines(close: pd.Series, fast_len: int, slow_len: int, signal_len: int, signal_sma: bool=False):
    fast = ema(close, fast_len)
    slow = ema(close, slow_len)
    macd_line = fast - slow
    if signal_sma:
        signal_line = sma(macd_line, signal_len)
    else:
        signal_line = ema(macd_line, signal_len)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def rolling_linreg_last_y(series: pd.Series, length: int) -> pd.Series:
    x = np.arange(length)
    sum_x = x.sum()
    sum_x2 = (x**2).sum()
    denom = (length * sum_x2 - sum_x**2)
    def _calc(win: pd.Series):
        y = win.values
        sum_y = y.sum()
        sum_xy = (x * y).sum()
        m = (length * sum_xy - sum_x * sum_y) / denom
        b = (sum_y - m * sum_x) / length
        return b + m * (length - 1)
    return series.rolling(window=length, min_periods=length).apply(_calc, raw=False)

def squeeze_momentum_lazybear(close, high, low, sqz_len, sqz_mult, kc_len, kc_mult, use_tr=True):
    basis = sma(close, sqz_len)
    dev = sqz_mult * close.rolling(sqz_len, min_periods=sqz_len).std()
    ma = sma(close, kc_len)
    rng = true_range(high, low, close) if use_tr else (high - low)
    rangema = sma(rng, kc_len)
    upperKC = ma + rangema * kc_mult
    lowerKC = ma - rangema * kc_mult
    upperBB = basis + dev
    lowerBB = basis - dev
    midKC = (high.rolling(kc_len, min_periods=kc_len).max() + low.rolling(kc_len, min_periods=kc_len).min()) / 2.0
    basisKC = sma(close, kc_len)
    avgValue = (midKC + basisKC) / 2.0
    val_input = close - avgValue
    val = rolling_linreg_last_y(val_input, kc_len)
    sqzOn = (lowerBB > lowerKC) & (upperBB < upperKC)
    sqzOff = (lowerBB < lowerKC) & (upperBB > upperKC)
    noSqz = ~(sqzOn | sqzOff)
    return val, sqzOn, sqzOff, noSqz

def fetch_ohlcv_df(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
    df = df.astype({"time":"int64","open":"float","high":"float","low":"float","close":"float","volume":"float"})
    df["dt"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df

def fetch_last_candle_df(symbol: str, timeframe: str) -> pd.DataFrame:
    return fetch_ohlcv_df(symbol, timeframe, limit=1)

def upsert_last_candle(df_all: pd.DataFrame, last_df: pd.DataFrame) -> pd.DataFrame:
    if last_df is None or len(last_df) == 0:
        return df_all
    last_new = last_df.iloc[-1]
    if len(df_all) == 0:
        return last_df.copy()
    last_time_all = int(df_all.iloc[-1]["time"])
    if int(last_new["time"]) > last_time_all:
        df_all = pd.concat([df_all, last_df], ignore_index=True)
    else:
        cols = ["time", "open", "high", "low", "close", "volume", "dt"]
        df_all.loc[df_all.index[-1], cols] = last_new[cols].values
    return df_all

def compute_outHist_mtf_from_intraday(df15: pd.DataFrame) -> pd.Series:
    df = df15.copy()
    df["day"] = df["dt"].dt.floor("D")
    day_close = df.groupby("day")["close"].last()
    ema_fast_day_end = day_close.ewm(span=fastLength_mtf, adjust=False).mean()
    ema_slow_day_end = day_close.ewm(span=slowLength_mtf, adjust=False).mean()
    macd_day_end = ema_fast_day_end - ema_slow_day_end
    prev_sum_Nm1 = macd_day_end.shift(1).rolling(signalLength_mtf-1, min_periods=signalLength_mtf-1).sum()
    prev_day = df["day"] - pd.Timedelta(days=1)
    prev_fast = prev_day.map(ema_fast_day_end)
    prev_slow = prev_day.map(ema_slow_day_end)
    alpha_fast = 2.0/(fastLength_mtf+1.0)
    alpha_slow = 2.0/(slowLength_mtf+1.0)
    ema_fast_now = alpha_fast * df["close"] + (1.0 - alpha_fast) * prev_fast
    ema_slow_now = alpha_slow * df["close"] + (1.0 - alpha_slow) * prev_slow
    macd_now = ema_fast_now - ema_slow_now
    prev_sum_for_bar = df["day"].map(prev_sum_Nm1)
    signal_now = (prev_sum_for_bar + macd_now) / signalLength_mtf
    outHist_now = macd_now - signal_now
    return outHist_now

def compute_indicators(df15: pd.DataFrame) -> pd.DataFrame:
    df15["rsi"] = rsi(df15["close"], rsi_length)
    macd_line, signal_line, _ = macd_lines(df15["close"], macd_fast_length, macd_slow_length, macd_signal_length, signal_sma=False)
    df15["macd_line"] = macd_line
    df15["signal_line"] = signal_line
    plusDI, minusDI, adx_value_series = adx_plus_minus_di(df15["high"], df15["low"], df15["close"], adx_length, adx_smoothing)
    df15["plusDI"] = plusDI
    df15["minusDI"] = minusDI
    df15["adx_value"] = adx_value_series
    val, _, _, _ = squeeze_momentum_lazybear(df15["close"], df15["high"], df15["low"], sqz_length, sqz_mult, kc_length, kc_mult, use_tr=True)
    df15["val"] = val
    df15["outHist"] = compute_outHist_mtf_from_intraday(df15)
    return df15

def build_conditions(df: pd.DataFrame) -> pd.DataFrame:
    cond1_long = (df["rsi"] < rsi_buy)
    cond2_long = (df["macd_line"] < df["signal_line"]) & (df["macd_line"] < -macd_threshold)
    cond3_long = (df["plusDI"] < df["minusDI"]) & (df["adx_value"] > adx_val)
    cond4_long = (df["val"] < sqzbuy)
    count_long = cond1_long.astype(int) + cond2_long.astype(int) + cond3_long.astype(int) + cond4_long.astype(int)
    df["long_condition"] = (count_long >= countbc) & (df["outHist"] < mtf_buy_threshold)
    cond1_short = (df["rsi"] > rsi_sell)
    cond2_short = (df["macd_line"] > df["signal_line"]) & (df["macd_line"] > macd_threshold)
    cond3_short = (df["plusDI"] > df["minusDI"]) & (df["adx_value"] > adx_val)
    cond4_short = (df["val"] > sqzsell)
    count_short = cond1_short.astype(int) + cond2_short.astype(int) + cond3_short.astype(int) + cond4_short.astype(int)
    df["short_condition"] = (count_short >= countbc) & (df["outHist"] > mtf_sell_threshold)
    sig = np.full(len(df), None, dtype=object)
    sig = np.where(df["long_condition"], "BUY", sig)
    sig = np.where(df["short_condition"], "SELL", sig)
    df["signal"] = pd.Series(sig, index=df.index, dtype=object)
    return df

def fmt_num(x, digits=2):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)

def ts_local_from_ms(ms: int) -> str:
    dt_local = datetime.fromtimestamp(ms/1000.0, tz=timezone.utc).astimezone(LOCAL_TZ)
    return dt_local.strftime("%Y/%m/%d ساعت: %H:%M")

# ---------- فرمت پیام تلگرام ----------
TELEGRAM_TOKEN = "8196141905:AAESgGc3lSVsO5qMGpm58QyuN2djifz3GGQ"
TELEGRAM_CHAT_ID = "-1003027394842"  # VIP
TELEGRAM_CHAT_ID_PUBLIC = "-1002419973211"  # عمومی

def send_telegram_message(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            print(f"❌ خطا در ارسال پیام تلگرام: {r.text}")
    except Exception as e:
        print(f"❌ خطا در ارتباط با تلگرام: {e}")

def send_telegram_message_to(message: str, chat_id: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            print(f"❌ خطا در ارسال پیام تلگرام: {r.text}")
    except Exception as e:
        print(f"❌ خطا در ارتباط با تلگرام: {e}")

def last_n_signals(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return df[df["signal"].notna()].tail(n)

# ---------- پیام سیگنال نسخه قدیم (حذف نشده ولی دیگر استفاده نمی‌شود) ----------
def format_signal_message(row: pd.Series, current_price: float) -> str:
    t_loc = ts_local_from_ms(int(row["time"]))
    kind = str(row["signal"])
    entry_price = float(row["close"])
    head_emoji = "🚀" if kind=="BUY" else "🧘‍♂️" if kind=="SELL" else "ℹ️"
    msg = (
        f"⏰ {t_loc}\n\n"
        f"close All old position \n new position started \n\n"
        f"{head_emoji} | {kind} \n\n"
        f"💸| price={fmt_num(entry_price,2)} \n"
        f".\n"
        f"| RSI={fmt_num(row['rsi'],2)} | ADX={fmt_num(row['adx_value'],2)} | MACD={fmt_num(row['macd_line'],2)}\n"
        f".\n"
        f"با حوصله در کمترین قیمت وارد شوید"
    )
    return msg

# =============== تغییرات جدید برای ساخت پیام طبق درخواست ===============

# تعیین فرمت اعشار DCA
def dca_price_str(kind: str, entry_price: float, k: int) -> str:
    # k = 0..9
    if kind == "SELL":
        level_price = entry_price * (1 + 0.01 * k)
        digits = 2 if k == 7 else 0
    else:
        level_price = entry_price * (1 - 0.01 * k)
        digits = 2 if k in (3, 7) else 0
    return fmt_num(level_price, digits)

def build_dca_lines(kind: str, entry_price: float):
    lines = []
    for k in range(10):
        price_s = dca_price_str(kind, entry_price, k)
        lines.append(f"10 درصد از موجودی از موجودی را در  price={price_s}  وارد شوید")
    return lines

def format_signal_message_vip(row: pd.Series, current_price: float) -> str:
    # پیام VIP با اینتر بجای نقطه، مطابق الگو و قواعد اعشار
    t_loc = ts_local_from_ms(int(row["time"]))
    kind = str(row["signal"])
    entry_price = float(row["close"])
    head_emoji = "🚀" if kind == "BUY" else "🧘‍♂️"

    lines = [
        "",
        "",
        f"⏰ {t_loc}",
        "",
        "close All old position ",
        " new position started ",
        "",
        f"{head_emoji} | {kind} ",
        "",
        f"💸| price={fmt_num(entry_price,2)} ",
        "",
        "",
        "",
    ]
    lines += build_dca_lines(kind, entry_price)
    lines += [
        "",
        f"| RSI={fmt_num(row['rsi'],2)} | ADX={fmt_num(row['adx_value'],2)} | MACD={fmt_num(row['macd_line'],2)}",
        "",
        "",
    ]
    return "\n".join(lines)

def format_signal_message_public(row: pd.Series, current_price: float) -> str:
    base = format_signal_message_vip(row, current_price)
    extra = [
        "این سیگنال در کانال عمومی با تاخیر ارسال شده.",
        "جهت دریافت در لحظه ی سیگنال به کانال VIP بپیوندید.",
        "موقتا عضویت در کانال VIP رایگان است.",
        "",
        "ارتباط با پشتیبانی:",
        "@btctrader321",
    ]
    return base + "\n" + "\n".join(extra)

# ---------- چاپ در کنسول ----------
def print_signal_console(prefix: str, row: pd.Series):
    t_loc = ts_local_from_ms(int(row["time"]))
    print(f"{prefix} {t_loc} | {row['signal']} | price={fmt_num(row['close'],2)} | RSI={fmt_num(row['rsi'],2)} | ADX={fmt_num(row['adx_value'],2)} | MACD={fmt_num(row['macd_line'],2)}")

# ==============================
# 4) MAIN
# ==============================
def main():
    df15 = fetch_ohlcv_df(symbol, tf, limit_15m)

    # ارسال پیام شروع
    startup_msg = "⏰ سیستم در حال بروزرسانی — وقت بخیر 🌞"
    send_telegram_message(startup_msg)
    send_telegram_message_to(startup_msg, TELEGRAM_CHAT_ID_PUBLIC)

    df15 = compute_indicators(df15)
    df15 = build_conditions(df15)

    # ۵ سیگنال آخر
    df15_closed_hist = df15.iloc[:-1].copy() if len(df15)>=2 else df15.copy()
    sigs = last_n_signals(df15_closed_hist,5)
    if len(sigs) == 0:
        print("ℹ️ هیچ سیگنال تاریخی یافت نشد.")
    else:
        print("🕔 ۵ سیگنال آخر تاریخچه:")
        for _, row in sigs.iterrows():
            print_signal_console("•", row)

    last_closed_row = df15.iloc[-2] if len(df15)>=2 else df15.iloc[-1]
    if pd.notna(last_closed_row["signal"]):
        preview_msg = format_signal_message_vip(last_closed_row, float(df15.iloc[-1]["close"]))  # پیش‌نمایش VIP
        print("ℹ️ سیگنال آخرین کندلِ بسته‌شده:\n", preview_msg)

    # محدودیت ارسال: تا تغییر جهت، سیگنال هم‌جهت دوباره ارسال نشود
    last_sent_signal = None  # None, "BUY", "SELL"

    # مانیتورینگ زنده
    while True:
        last_candle = fetch_last_candle_df(symbol, tf)
        df15 = upsert_last_candle(df15, last_candle)
        df15 = compute_indicators(df15)
        df15 = build_conditions(df15)
        current_row = df15.iloc[-1]

        if pd.notna(current_row["signal"]):
            current_sig = str(current_row["signal"])
            if last_sent_signal != current_sig:
                # VIP
                msg_vip = format_signal_message_vip(current_row, float(current_row["close"]))
                print("📢 سیگنال جدید (VIP):\n", msg_vip)
                send_telegram_message(msg_vip)

                # عمومی با تاخیر
                msg_public = format_signal_message_public(current_row, float(current_row["close"]))
                def send_public_delayed():
                    send_telegram_message_to(msg_public, TELEGRAM_CHAT_ID_PUBLIC)
                threading.Timer(delay_public_seconds, send_public_delayed).start()

                # به‌روزرسانی جهت آخرین سیگنال
                last_sent_signal = current_sig
            else:
                # همان جهت تکراری: ارسال نمی‌شود
                pass

        time.sleep(poll_seconds)

if __name__ == "__main__":
    main()
