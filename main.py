# -*- coding: utf-8 -*-
import time
from datetime import datetime, timezone, timedelta

import ccxt
import numpy as np
import pandas as pd
import requests

# ==============================
# 0) MINIMAL WEB SERVER (Render keeps the service alive)
# ==============================
import os
import threading
try:
    from flask import Flask, jsonify
    _flask_available = True
except Exception:
    # اگر Flask نصب نبود، برنامه همچنان کار می‌کند؛ فقط بخش وب غیرفعال می‌شود
    _flask_available = False

PORT = int(os.environ.get("PORT", "10000"))
# Render مقدار RENDER_EXTERNAL_URL را ست می‌کند. اگر نبود، می‌توانید SELF_URL را دستی ست کنید
SELF_URL = os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("SELF_URL")

if _flask_available:
    app = Flask(__name__)

    @app.get("/")
    def root():
        return jsonify({
            "ok": True,
            "service": "signals",
            "time": datetime.utcnow().isoformat() + "Z"
        })

    @app.get("/health")
    def health():
        return "ok", 200

    def _run_web():
        try:
            # روی پورت ارائه‌شده توسط Render گوش می‌دهد
            app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
        except Exception as e:
            print(f"⚠️ Flask web server failed: {e}")
else:
    def _run_web():
        # Flask موجود نیست → چیزی برای اجرا نیست
        pass


def _keep_alive_pinger():
    """
    پینگ دوره‌ای به آدرس سرویس (برای بیدار نگه داشتن Render Free Web Service)
    اگر RENDER_EXTERNAL_URL یا SELF_URL ست نشده باشد، این بخش صرفاً پیام اطلاع می‌دهد و رد می‌شود.
    """
    if not SELF_URL:
        print("ℹ️ RENDER_EXTERNAL_URL/SELF_URL not set; skipping self-ping.")
        return
    url = SELF_URL.rstrip("/") + "/health"
    print(f"⏳ keep-alive ping thread → {url}")
    while True:
        try:
            requests.get(url, timeout=10)
        except Exception as e:
            print(f"⚠️ keep-alive ping failed: {e}")
        time.sleep(240)  # هر 4 دقیقه یکبار


def start_keep_alive_webserver():
    try:
        threading.Thread(target=_run_web, daemon=True).start()
        threading.Thread(target=_keep_alive_pinger, daemon=True).start()
    except Exception as e:
        print(f"⚠️ keep-alive bootstrap failed: {e}")










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
adx_smoothing = length  # در Pine از RMA هم برای DX هم برای ADX استفاده می‌شود

# TP/SL (برای تطابق، در منطق سیگنال استفاده نمی‌شود)
tp_percent = 2.0
sl_percent = 1.0

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
useCurrentRes = False           # مثل Pine: پیش‌فرض false
resCustom = "1d"                # HTF
mtf_buy_threshold = -700.0
mtf_sell_threshold = 700.0
fastLength_mtf = 12
slowLength_mtf = 26
signalLength_mtf = 9           # توجه: سیگنالِ MTF = SMA روی مکدی (مثل Pine)

# ==============================
# 2) EXCHANGE / DATA SETTINGS
# ==============================
symbol = "BTC/USDT"       # KuCoin
tf = "15m"                # تایم‌فریم اصلی
limit_15m = 5000           # تعداد کندل‌ها
poll_seconds = 300         # فاصله مانیتورینگ زنده (ثانیه) ← هر ۵ دقیقه
LIVE = True               # برای مانیتورینگ زنده True/False

# --- تاخیر ارسال به کانال عمومی بر حسب ساعت (قابل تنظیم)
delay_public_hours = 1.0
delay_public_seconds = int(delay_public_hours * 3600)

# ناحیه زمانی محلی (UTC+3:30)
LOCAL_TZ = timezone(timedelta(hours=3, minutes=30))

# اتصال KuCoin (از پروکسی سیستم استفاده می‌کند؛ چیزی در کد ست نشده)
exchange = ccxt.kucoin({
    "timeout": 60000,
    "enableRateLimit": True,
})

# ==============================
# 3) HELPER FUNCTIONS
# ==============================

def rma(series: pd.Series, period: int) -> pd.Series:
    """RMA (Wilder) دقیقاً مثل ta.rma"""
    return series.ewm(alpha=1.0/period, adjust=False).mean()

def rsi(series: pd.Series, period: int) -> pd.Series:
    """RSI = ta.rsi"""
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
    """
    +DI / -DI و ADX به سبک Pine
    """
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
    """EMA با adjust=False مثل Pine"""
    return series.ewm(span=length, adjust=False).mean()

def sma(series: pd.Series, length: int) -> pd.Series:
    """SMA استاندارد مثل ta.sma"""
    return series.rolling(window=length, min_periods=length).mean()

def macd_lines(close: pd.Series, fast_len: int, slow_len: int, signal_len: int, signal_sma: bool=False):
    """
    MACD = EMA(fast) - EMA(slow)
    signal_sma=True برای MTF (مانند Pine → SMA)
    """
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
    """
    معادل ta.linreg(series, length, 0)
    """
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
    """
    پیاده‌سازی LazyBear Squeeze Momentum مطابق Pine
    """
    basis = sma(close, sqz_len)
    dev = sqz_mult * close.rolling(sqz_len, min_periods=sqz_len).std()
    ma = sma(close, kc_len)
    rng = true_range(high, low, close) if use_tr else (high - low)
    rangema = sma(rng, kc_len)

    upperKC = ma + rangema * kc_mult
    lowerKC = ma - rangema * kc_mult

    upperBB = basis + dev
    lowerBB = basis - dev

    midKC = (high.rolling(kc_len, min_periods=kc_len).max() +
             low.rolling(kc_len, min_periods=kc_len).min()) / 2.0
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

# ---- فقط کندل آخر را بگیر
def fetch_last_candle_df(symbol: str, timeframe: str) -> pd.DataFrame:
    """فقط آخرین کندل 15m را از API می‌گیرد"""
    return fetch_ohlcv_df(symbol, timeframe, limit=1)

# ---- ادغام/به‌روزرسانی کندل آخر در دیتافریم اصلی
def upsert_last_candle(df_all: pd.DataFrame, last_df: pd.DataFrame) -> pd.DataFrame:
    """
    اگر time کندل آخر جدیدتر از df_all باشد → append
    اگر همان کندلِ در حال تشکیل باشد → سطر آخر را آپدیت می‌کنیم
    """
    if last_df is None or len(last_df) == 0:
        return df_all
    last_new = last_df.iloc[-1]
    if len(df_all) == 0:
        return last_df.copy()

    last_time_all = int(df_all.iloc[-1]["time"])
    if int(last_new["time"]) > last_time_all:
        # کندل جدید
        df_all = pd.concat([df_all, last_df], ignore_index=True)
    else:
        # همان کندل جاری → آپدیت سطر آخر
        cols = ["time", "open", "high", "low", "close", "volume", "dt"]
        df_all.loc[df_all.index[-1], cols] = last_new[cols].values
    return df_all

# ---- MTF: مشابه request.security(..., "1D", hist_mtf) با مقدار روز جاری ----
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
    # RSI
    df15["rsi"] = rsi(df15["close"], rsi_length)

    # MACD تایم‌فریم اصلی (signal = EMA)
    macd_line, signal_line, _ = macd_lines(
        df15["close"], macd_fast_length, macd_slow_length, macd_signal_length, signal_sma=False
    )
    df15["macd_line"] = macd_line
    df15["signal_line"] = signal_line

    # ADX + +DI/-DI
    plusDI, minusDI, adx_value_series = adx_plus_minus_di(
        df15["high"], df15["low"], df15["close"], adx_length, adx_smoothing
    )
    df15["plusDI"] = plusDI
    df15["minusDI"] = minusDI
    df15["adx_value"] = adx_value_series

    # SQZ LazyBear (val)
    val, _, _, _ = squeeze_momentum_lazybear(
        df15["close"], df15["high"], df15["low"],
        sqz_length, sqz_mult, kc_length, kc_mult, use_tr=True
    )
    df15["val"] = val

    # MTF هیستوگرام روزانه (مثل Pine + مقدار روز جاری)
    df15["outHist"] = compute_outHist_mtf_from_intraday(df15)
    return df15

def build_conditions(df: pd.DataFrame) -> pd.DataFrame:
    # LONG
    cond1_long = (df["rsi"] < rsi_buy)
    cond2_long = (df["macd_line"] < df["signal_line"]) & (df["macd_line"] < -macd_threshold)
    cond3_long = (df["plusDI"] < df["minusDI"]) & (df["adx_value"] > adx_val)
    cond4_long = (df["val"] < sqzbuy)
    count_long = cond1_long.astype(int) + cond2_long.astype(int) + cond3_long.astype(int) + cond4_long.astype(int)
    df["long_condition"] = (count_long >= countbc) & (df["outHist"] < mtf_buy_threshold)

    # SHORT
    cond1_short = (df["rsi"] > rsi_sell)
    cond2_short = (df["macd_line"] > df["signal_line"]) & (df["macd_line"] > macd_threshold)
    cond3_short = (df["plusDI"] > df["minusDI"]) & (df["adx_value"] > adx_val)
    cond4_short = (df["val"] > sqzsell)
    count_short = cond1_short.astype(int) + cond2_short.astype(int) + cond3_short.astype(int) + cond4_short.astype(int)
    df["short_condition"] = (count_short >= countbc) & (df["outHist"] > mtf_sell_threshold)

    # --- سیگنال نهایی با dtype=object برای جلوگیری از DTypePromotionError ---
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
    """برگرداندن رشته زمان در UTC+3:30 از epoch میلی‌ثانیه"""
    dt_local = datetime.fromtimestamp(ms/1000.0, tz=timezone.utc).astimezone(LOCAL_TZ)
    return dt_local.strftime("%Y-%m-%d %H:%M")

# ---------- فرمت دقیق پیام تلگرام مطابق نمونه ----------
ICONS = {1: "1️⃣", 3: "3️⃣", 5: "5️⃣", 7: "7️⃣", 10: "🔟"}

def format_signal_message(row: pd.Series, current_price: float) -> str:
    t_loc = ts_local_from_ms(int(row["time"]))
    kind = str(row["signal"])
    entry_price = float(row["close"])

    # محاسبه سود درصدی (روی قیمت فعلی)
    if kind == "BUY":
        profit_pct = (current_price - entry_price) / entry_price * 100.0
        head_emoji = "🚀"
    elif kind == "SELL":
        profit_pct = (entry_price - current_price) / entry_price * 100.0
        head_emoji = "🧘‍♂️"
    else:
        profit_pct = 0.0
        head_emoji = "ℹ️"

    # خطوط سود برای لوریج‌های مختلف (1x, 3x, 5x, 7x, 10x)
    leverages = [1, 3, 5, 7, 10]
    profit_lines = []
    for lev in leverages:
        lev_icon = ICONS.get(lev, "")
        profit_lines.append(f"{lev_icon}profit {lev}x={fmt_num(profit_pct*lev, 2)}%")
    profits_block = "\n".join(profit_lines)

    # بلوک اندیکاتورها
    indi_block = (
        f"| RSI={fmt_num(row['rsi'],2)} | ADX={fmt_num(row['adx_value'],2)} "
        f"| MACD={fmt_num(row['macd_line'],2)} | SQZ={fmt_num(row['val'],2)} | MTF={fmt_num(row['outHist'],2)}"
    )

    # ساخت پیام نهایی مطابق نمونهٔ کاربر
    msg = (
        f"⏰{t_loc}\n"
        f"{head_emoji} | {kind} \n"
        f"💸| price={fmt_num(entry_price,2)} \n"
        f".\n"
        f"{indi_block}\n"
        f".\n"
        f"{profits_block}"
    )
    return msg

# ---------- ارسال تلگرام ----------
TELEGRAM_TOKEN = "8196141905:AAESgGc3lSVsO5qMGpm58QyuN2djifz3GGQ"

TELEGRAM_CHAT_ID = "-1003027394842"         # کانال VIP (اصلی - ارسال فوری)
TELEGRAM_CHAT_ID_PUBLIC = "-1002419973211"  # کانال عمومی (ارسال با تاخیر برای سیگنال)

def send_telegram_message(message: str):
    """ارسال به کانال VIP (سازگاری با نسخه قبلی)"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "disable_web_page_preview": True
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            print(f"❌ خطا در ارسال پیام تلگرام: {r.text}")
    except Exception as e:
        print(f"❌ خطا در ارتباط با تلگرام: {e}")

def send_telegram_message_to(message: str, chat_id: str):
    """ارسال به هر چت آی‌دی دلخواه (برای کانال عمومی/سایر کانال‌ها)"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "disable_web_page_preview": True
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            print(f"❌ خطا در ارسال پیام تلگرام: {r.text}")
    except Exception as e:
        print(f"❌ خطا در ارتباط با تلگرام: {e}")

def last_n_signals(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return df[df["signal"].notna()].tail(n)

# Helper: print last 5 signals to console only (bullet lines, no headers)  <<< ADDED (console 5-signals)

def print_last_five_signals_console_only(df: pd.DataFrame):
    try:
        if len(df) == 0:
            return
        current_price = float(df.iloc[-1]["close"])  # قیمت لحظه‌ای از کندل جاری
        df_closed_hist = df.iloc[:-1].copy() if len(df) >= 2 else df.copy()
        sigs = last_n_signals(df_closed_hist, 5)
        if len(sigs) == 0:
            return
        # چاپ فقط خطوط بولت مانند نمونه کاربر
        for _, row in sigs.iterrows():
            print_signal_with_profit("•", row, current_price)
    except Exception as _e:
        # فقط لاگ نرم
        print(f"⚠️ print_last_five_signals_console_only error: {_e}")


# ==============================
# 4) MAIN
# ==============================

def print_signal_with_profit(prefix: str, row: pd.Series, current_price: float):
    """چاپ سیگنال تاریخچه با سود فعلی (UTC+3:30) در کنسول"""
    kind = row["signal"]
    entry = float(row["close"])
    if kind == "BUY":
        profit_pct = (current_price - entry) / entry * 100.0
    elif kind == "SELL":
        profit_pct = (entry - current_price) / entry * 100.0
    else:
        profit_pct = 0.0
    t_loc = ts_local_from_ms(int(row["time"]))
    print(
        f"{prefix} {t_loc} | {kind} | price={fmt_num(entry,2)} "
        f"| RSI={fmt_num(row['rsi'],2)} | ADX={fmt_num(row['adx_value'],2)} "
        f"| MACD={fmt_num(row['macd_line'],2)} | SQZ={fmt_num(row['val'],2)} | MTF={fmt_num(row['outHist'],2)} "
        f"| profit={fmt_num(profit_pct,2)}%"
    )

def main():
    # 1) داده‌ها
    df15 = fetch_ohlcv_df(symbol, tf, limit_15m)

    # پیام تست شروع (همزمان به VIP و PUBLIC)
    startup_msg = "⏰ برنامه شروع شد — وقت بخیر 🌞"
    send_telegram_message(startup_msg)  # VIP
    send_telegram_message_to(startup_msg, TELEGRAM_CHAT_ID_PUBLIC)  # PUBLIC

    # 2) اندیکاتورها + شرایط
    df15 = compute_indicators(df15)
    df15 = build_conditions(df15)

    # 3) چاپ ۵ سیگنال آخر تاریخچه (با قیمت و سود تا این لحظه)
    # --- فقط کندل‌های بسته‌شده را در تاریخچه در نظر بگیر (آخرین سطر معمولاً کندل جاری است)  <<< CHANGED
    df15_closed_hist = df15.iloc[:-1].copy() if len(df15) >= 2 else df15.copy()  # <<< CHANGED
    sigs = last_n_signals(df15_closed_hist, 5)  # <<< CHANGED
    if len(sigs) == 0:
        print("ℹ️ هیچ سیگنال تاریخی یافت نشد.")
    else:
        current_price = float(df15.iloc[-1]["close"])  # قیمت لحظه‌ای از کندل جاری
        print("🕔 ۵ سیگنال آخر تاریخچه (UTC+3:30) با سود فعلی (مبنای ورود = کندل‌های بسته‌شده):")  # <<< CHANGED
        for _, row in sigs.iterrows():
            print_signal_with_profit("•", row, current_price)

    # 4) اگر آخرین کندلِ بسته‌شده سیگنال دارد، در کنسول اطلاع بده (پیش‌نمایش)  <<< CHANGED
    last_closed_row = df15.iloc[-2] if len(df15) >= 2 else df15.iloc[-1]  # <<< CHANGED
    if pd.notna(last_closed_row["signal"]):  # <<< CHANGED
        current_price = float(df15.iloc[-1]["close"])  # قیمت لحظه‌ای
        preview_msg = format_signal_message(last_closed_row, current_price)  # <<< CHANGED
        print("ℹ️ سیگنال آخرین کندلِ بسته‌شده:\n" + preview_msg)  # <<< CHANGED

    # 5) مانیتورینگ زنده: فقط وقتی «سیگنال جدید روی کندل بسته‌شده» صادر شد ارسال/چاپ کن  <<< CHANGED
    last_printed_time = int(last_closed_row["time"]) if len(df15) else 0  # <<< CHANGED

    if not LIVE:
        return

    while True:
        try:
            # فقط کندل آخر را از API بگیر و به df اضافه/به‌روز کن
            last_df = fetch_last_candle_df(symbol, tf)
            df15 = upsert_last_candle(df15, last_df)

            # برای ارزیابی سیگنال روی کندلِ بسته‌شده، حداقل به 3 سطر نیاز داریم: [..., prev_closed, last_closed, current_open]  <<< CHANGED
            if len(df15) < 3:
                time.sleep(poll_seconds)
                continue

            # اندیکاتورها و شرایط را مجدداً محاسبه کن
            df15 = compute_indicators(df15)
            df15 = build_conditions(df15)

            # «کندل بسته‌شدهٔ اخیر» و «کندل بسته‌شدهٔ قبلی»  <<< CHANGED
            lr = df15.iloc[-2]   # last closed bar  <<< CHANGED
            pr = df15.iloc[-3]   # previous closed bar  <<< CHANGED
            sig = lr["signal"]
            prev_sig = pr["signal"] if "signal" in pr else None
            bar_time = int(lr["time"])  # زمان کندل بسته‌شدهٔ اخیر  <<< CHANGED

            # فقط وقتی کندل بسته‌شدهٔ جدید داریم، و سیگنال نسبت به کندل بسته‌شدهٔ قبلی تغییر کرده باشد  <<< CHANGED
            is_new_closed_bar = (bar_time != last_printed_time)  # <<< CHANGED
            is_new_signal_event = is_new_closed_bar and pd.notna(sig) and ( (not pd.notna(prev_sig)) or (sig != prev_sig) )  # <<< CHANGED

            if is_new_signal_event:
                current_price = float(df15.iloc[-1]["close"])  # قیمت لحظه‌ای از کندل جاری
                message_text = format_signal_message(lr, current_price)  # مبنا = کندل بسته‌شده  <<< CHANGED

                # چاپ در کنسول
                print("📢 سیگنال جدید (کندل بسته‌شده):\n" + message_text)  # <<< CHANGED

                # ارسال فوری به کانال VIP (اصلی)
                send_telegram_message(message_text)

                # آماده‌سازی پیام کانال عمومی با متن انتهایی
                delayed_msg = (
                    message_text +
                    "\n\n"
                    "این سیگنال در کانال عمومی با تاخیر ارسال شده\n"
                    "جهت دریافت در لحظه ی سیگنال به کانال VIP بپیوندید\n"
                    "موقتا عضویت در کانال VIP رایگان است\n\n"
                    "ارتباط با پشتیبانی:\n"
                    "@btctrader321\n"
                )

                # ارسال به کانال عمومی فقط برای سیگنال جدید، با تاخیر تنظیم‌شده
                def send_delayed():
                    time.sleep(delay_public_seconds)  # تاخیر بر حسب ثانیه (از ساعت محاسبه شده)
                    send_telegram_message_to(delayed_msg, TELEGRAM_CHAT_ID_PUBLIC)

                import threading
                threading.Thread(target=send_delayed, daemon=True).start()

                # بعد از ارسال، ۵ سیگنال آخر فقط در کنسول چاپ شود (فرمت بولت)
                print_last_five_signals_console_only(df15)  # <<< ADDED (console 5-signals)

                # به‌روزرسانی لاک با «زمان کندل بسته‌شدهٔ ارسال‌شده»  <<< CHANGED
                last_printed_time = bar_time  # <<< CHANGED

        except Exception as e:
            print(f"❌ خطا: {e}")

        time.sleep(poll_seconds)

if __name__ == "__main__":
    # راه‌اندازی وب‌سرور و پینگِ نگهدارنده (بدون تغییر در منطق ارسال تلگرام)
    start_keep_alive_webserver()
    main()
