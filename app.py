"""
Order Book Footprint Backend + Order Flow X-Ray + Combined Signal Strategy + GEX Engine
Run:
    pip install fastapi uvicorn websockets requests aiohttp matplotlib mplfinance pandas
    uvicorn app:app --port 8000

FIXES IN THIS VERSION:
    1. http_get_json now retries (with backoff) on failure/429/418 instead of
       silently returning None on the first hiccup. Combined with...
    2. ...a shared short-TTL cache (cached_http_get_json) used for the heavy,
       frequently-polled Binance endpoints (futures order book depth, ticker
       price/24hr, open interest, premium index). Previously /api/liquidity_magnet
       and /api/liquidity_zones each independently fetched the SAME
       depth?limit=1000 call (the heaviest-weight endpoint) every few seconds,
       and market_metrics fetched several endpoints sequentially every 5s with
       no retry - together this was enough traffic to occasionally trip
       Binance's weight-based rate limit, which silently returned None and
       showed up as price/OI/funding flickering to 0. Now duplicate calls
       within the cache window are served from cache, and a transient failure
       gets retried instead of immediately giving up.
    3. Spoofing detector rewritten - it was structurally unable to ever emit
       an event:
         a) The trade-tape cross-check used a 0.05% price tolerance, which on
            BTC (~$90k) is a ~$45 band - trades happen in that band constantly,
            so the check almost always found a "real trade" nearby and
            discarded every candidate, spoof or not.
         b) The old logic set `partially_filled = True` forever on ANY size
            decrease at a price level. But a depth level is the SUM of every
            trader's resting orders at that price, not one single order - an
            unrelated trader nudging their own order at the same price would
            permanently disqualify a genuine large spoof sitting there too.
       Fixed by: using a much tighter price tolerance, and replacing the
       boolean "any trade nearby = discard" check with a volume-based one -
       only discard a candidate if the ACTUAL traded volume near that price
       during the window covers a meaningful fraction of the size that
       vanished (i.e. it was genuinely filled), not just because some
       unrelated trade happened to print nearby.
    4. Liquidity Magnet & Liquidity Zones now only consider walls >= $10M.
       Score tiers: $50M+ => 99, $25M+ => 85, $15M+ => 70, $10M+ => 55.
"""

import asyncio
import json
import time
import math
import os
import ssl
import datetime
from collections import defaultdict, deque

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import websockets
import requests
import aiohttp
from fastapi import FastAPI, APIRouter
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplfinance as mpf
    import pandas as pd
    SCREENSHOTS_AVAILABLE = True
except ImportError:
    SCREENSHOTS_AVAILABLE = False

_http_session = None


async def get_http_session():
    global _http_session
    if _http_session is None or _http_session.closed:
        _http_session = aiohttp.ClientSession()
    return _http_session


async def http_get_json(url, timeout=10, retries=2):
    """
    *** FIX: retry on transient failure / rate limiting ***
    Previously a single 429 or network hiccup made this return None
    immediately, which showed up in the UI as price/OI/funding flickering
    to zero. Now it retries a couple of times with a short backoff, with
    extra delay specifically for 429 (Too Many Requests) / 418 (IP ban)
    responses.
    """
    for attempt in range(retries + 1):
        try:
            session = await get_http_session()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status in (429, 418):
                    wait = 1.5 * (attempt + 1)
                    print(f"[HTTP] Rate limited ({resp.status}) on {url}, retrying in {wait:.1f}s...")
                    await asyncio.sleep(wait)
                    continue
                if resp.status != 200:
                    print(f"[HTTP] {resp.status} from {url}")
                    if attempt < retries:
                        await asyncio.sleep(0.5)
                        continue
                    return None
                return await resp.json()
        except Exception as e:
            print(f"[HTTP] GET failed (attempt {attempt + 1}/{retries + 1}) for {url}: {type(e).__name__}: {e}")
            if attempt < retries:
                await asyncio.sleep(0.5)
                continue
            return None
    return None


# ============================================================================
# *** FIX: shared short-TTL cache ***
# Multiple endpoints (liquidity_magnet, liquidity_zones, market_metrics) were
# each independently hitting the SAME heavy Binance endpoints every few
# seconds. This cache lets repeated calls to the same URL within a short
# window reuse one real fetch, cutting request volume (and rate-limit risk)
# substantially without adding noticeable staleness.
# ============================================================================
_shared_cache = {}
_shared_cache_lock = asyncio.Lock()


async def cached_http_get_json(url, ttl, timeout=8):
    now = time.time()
    async with _shared_cache_lock:
        entry = _shared_cache.get(url)
        if entry and (now - entry["ts"]) < ttl:
            return entry["data"]
    data = await http_get_json(url, timeout=timeout)
    if data is not None:
        async with _shared_cache_lock:
            _shared_cache[url] = {"data": data, "ts": now}
    return data


# ============================================================================
# GEX ENGINE
# ============================================================================
DERIBIT_BASE = "https://www.deribit.com/api/v2"
GEX_SUPPORTED_CURRENCIES = ["BTC", "ETH"]
GEX_CONTRACT_SIZE = {"BTC": 1.0, "ETH": 1.0}
GEX_RISK_FREE_RATE = 0.0
GEX_DIVIDEND_YIELD = 0.0
GEX_MIN_T_YEARS = 1.0 / (365 * 24)
GEX_MIN_SIGMA = 0.01
GEX_CONTRIBUTION_WALL_THRESHOLD = 5.0
GEX_CACHE_TTL_SECONDS = 180

_gex_cache = {}
_gex_cache_lock = asyncio.Lock()
gex_router = APIRouter()

_gex_session = None


def _get_gex_session():
    global _gex_session
    if _gex_session is None:
        _gex_session = requests.Session()
        _gex_session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
        })
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=4,
            pool_maxsize=4,
            max_retries=2,
        )
        _gex_session.mount("https://", adapter)
    return _gex_session


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x):
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def _gex_compute_d1(S, K, T, r, q, sigma):
    T = max(T, GEX_MIN_T_YEARS)
    sigma = max(sigma, GEX_MIN_SIGMA)
    return (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def _gex_compute_gamma(S, K, T, r, q, sigma):
    T = max(T, GEX_MIN_T_YEARS)
    sigma = max(sigma, GEX_MIN_SIGMA)
    d1 = _gex_compute_d1(S, K, T, r, q, sigma)
    return _norm_pdf(d1) / (S * sigma * math.sqrt(T))


async def _gex_fetch_json(url, params=None):
    def _sync_fetch():
        session = _get_gex_session()
        for verify_ssl in (True, False):
            try:
                resp = session.get(url, params=params, timeout=20, verify=verify_ssl)
                if resp.status_code == 429:
                    print(f"[GEX] Rate limited (429) on {url}")
                    return None
                if resp.status_code != 200:
                    print(f"[GEX] HTTP {resp.status_code} from {url} -> {resp.text[:200]}")
                    continue
                data = resp.json()
                return data.get("result")
            except Exception as e:
                print(f"[GEX] Fetch error (verify={verify_ssl}) for {url}: "
                      f"{type(e).__name__}: {e}")
        return None
    return await asyncio.to_thread(_sync_fetch)


async def _gex_fetch_book_summary(currency):
    url = f"{DERIBIT_BASE}/public/get_book_summary_by_currency"
    result = await _gex_fetch_json(url, {"currency": currency, "kind": "option"})
    return result or []


async def _gex_fetch_index_price(currency):
    url = f"{DERIBIT_BASE}/public/get_index_price"
    result = await _gex_fetch_json(url, {"index_name": f"{currency.lower()}_usd"})
    return result.get("index_price") if result else None


def _gex_parse_instrument(name):
    parts = name.split("-")
    if len(parts) != 4:
        return None, None, None
    _, expiry_str, strike_str, cp = parts
    try:
        strike = float(strike_str)
    except ValueError:
        return None, None, None
    return expiry_str, strike, ("call" if cp.upper() == "C" else "put")


def _gex_years_to_expiry(expiry_str, now_ts):
    try:
        expiry_date = datetime.datetime.strptime(expiry_str, "%d%b%y").replace(
            hour=8, minute=0, second=0, tzinfo=datetime.timezone.utc)
        now = datetime.datetime.fromtimestamp(now_ts, tz=datetime.timezone.utc)
        return max((expiry_date - now).total_seconds() / (365.0 * 24 * 3600), 0.0)
    except Exception:
        return 0.0


async def _gex_compute_snapshot(currency):
    currency = currency.upper()
    contract_size = GEX_CONTRACT_SIZE.get(currency, 1.0)
    now_ts = time.time()

    summaries = await _gex_fetch_book_summary(currency)
    if not summaries:
        return {"error": f"Could not fetch book summary for {currency}. Check server logs."}

    spot = await _gex_fetch_index_price(currency)
    if spot is None:
        return {"error": f"Could not fetch index price for {currency}. Check server logs."}

    strike_data = defaultdict(lambda: defaultdict(lambda: {
        "call_oi": 0.0, "put_oi": 0.0, "gex_call": 0.0, "gex_put": 0.0
    }))
    used_bs_count = 0
    skipped_no_iv = 0

    for item in summaries:
        name = item.get("instrument_name")
        if not name:
            continue
        expiry_str, strike, option_type = _gex_parse_instrument(name)
        if strike is None:
            continue

        oi = item.get("open_interest", 0.0) or 0.0
        if oi <= 0:
            continue

        mark_iv = item.get("mark_iv")
        if not mark_iv or mark_iv <= 0:
            skipped_no_iv += 1
            continue

        T = _gex_years_to_expiry(expiry_str, now_ts)
        gamma = _gex_compute_gamma(spot, strike, T, GEX_RISK_FREE_RATE,
            GEX_DIVIDEND_YIELD, mark_iv / 100.0)
        if gamma is None or gamma <= 0:
            continue
        used_bs_count += 1

        gex_contribution = gamma * oi * contract_size * (spot ** 2) * 0.01
        if option_type == "call":
            strike_data[expiry_str][strike]["call_oi"] += oi
            strike_data[expiry_str][strike]["gex_call"] += gex_contribution
        else:
            strike_data[expiry_str][strike]["put_oi"] += oi
            strike_data[expiry_str][strike]["gex_put"] += -gex_contribution

    combined_by_strike = defaultdict(float)
    per_expiry_totals = []
    strike_rows = []
    expiry_days_map = {
        e: round(_gex_years_to_expiry(e, now_ts) * 365, 1) for e in strike_data.keys()
    }
    for expiry_str, strikes in strike_data.items():
        expiry_total = 0.0
        for strike, d in strikes.items():
            gex_strike = d["gex_call"] + d["gex_put"]
            expiry_total += gex_strike
            combined_by_strike[strike] += gex_strike
            strike_rows.append({
                "expiry": expiry_str, "days_to_expiry": expiry_days_map[expiry_str],
                "strike": strike, "call_oi": d["call_oi"], "put_oi": d["put_oi"],
                "gex_call": round(d["gex_call"], 2), "gex_put": round(d["gex_put"], 2),
                "gex_strike": round(gex_strike, 2),
            })
        per_expiry_totals.append({
            "expiry": expiry_str, "days_to_expiry": expiry_days_map[expiry_str],
            "total_gex": round(expiry_total, 2),
        })
    per_expiry_totals.sort(key=lambda e: e["days_to_expiry"])

    total_gex = sum(combined_by_strike.values())
    total_absolute_gex = sum(abs(v) for v in combined_by_strike.values())
    condition = "STABLE" if total_gex > 0 else ("VOLATILE" if total_gex < 0 else "NEUTRAL")

    NEAR_TERM_DAYS_CUTOFF = 7
    near_term_by_strike = defaultdict(float)
    for expiry_str, strikes in strike_data.items():
        if expiry_days_map[expiry_str] <= NEAR_TERM_DAYS_CUTOFF:
            for strike, d in strikes.items():
                near_term_by_strike[strike] += d["gex_call"] + d["gex_put"]
    near_term_total_gex = sum(near_term_by_strike.values())
    near_term_total_absolute_gex = sum(abs(v) for v in near_term_by_strike.values())
    near_term_condition = (
        ("STABLE" if near_term_total_gex > 0 else "VOLATILE")
        if near_term_total_absolute_gex > 0 else "NO_NEAR_TERM_DATA"
    )

    def _detect_walls(by_strike_map, total_abs):
        found = []
        if total_abs <= 0:
            return found
        for strike, gex_strike in by_strike_map.items():
            pct = (abs(gex_strike) / total_abs) * 100.0
            if pct <= GEX_CONTRIBUTION_WALL_THRESHOLD:
                continue
            if strike < spot and gex_strike > 0:
                wt = "SUPPORT WALL"
            elif strike > spot and gex_strike > 0:
                wt = "RESISTANCE WALL"
            elif strike < spot and gex_strike < 0:
                wt = "DOWNSIDE BREAKOUT ZONE"
            elif strike > spot and gex_strike < 0:
                wt = "UPSIDE BREAKOUT ZONE"
            else:
                continue
            found.append({
                "strike": strike, "gex": round(gex_strike, 2),
                "contribution_pct": round(pct, 2), "type": wt,
            })
        found.sort(key=lambda w: w["contribution_pct"], reverse=True)
        return found

    zero_gamma_levels = []
    if total_absolute_gex > 0:
        sorted_strikes = sorted(combined_by_strike.keys())
        cumulative = 0.0
        prev_strike, prev_cumulative = None, None
        for strike in sorted_strikes:
            cumulative += combined_by_strike[strike]
            if prev_cumulative is not None and (
                (prev_cumulative < 0 <= cumulative) or (prev_cumulative > 0 >= cumulative)
            ):
                if cumulative != prev_cumulative:
                    frac = -prev_cumulative / (cumulative - prev_cumulative)
                    zero_gamma_levels.append(round(prev_strike + frac * (strike - prev_strike), 2))
            prev_strike, prev_cumulative = strike, cumulative

    primary_zero = (min(zero_gamma_levels, key=lambda lvl: abs(lvl - spot))
                    if zero_gamma_levels else None)

    return {
        "currency": currency, "spot": spot, "timestamp": int(now_ts * 1000),
        "combined": {
            "total_gex": round(total_gex, 2),
            "total_absolute_gex": round(total_absolute_gex, 2),
            "condition": condition,
            "walls": _detect_walls(combined_by_strike, total_absolute_gex),
        },
        "near_term": {
            "days_cutoff": NEAR_TERM_DAYS_CUTOFF,
            "total_gex": round(near_term_total_gex, 2),
            "total_absolute_gex": round(near_term_total_absolute_gex, 2),
            "condition": near_term_condition,
            "walls": _detect_walls(near_term_by_strike, near_term_total_absolute_gex),
        },
        "zero_gamma_levels": zero_gamma_levels,
        "primary_zero_gamma_level": primary_zero,
        "per_expiry_totals": per_expiry_totals,
        "strike_breakdown": sorted(strike_rows, key=lambda r: (r["days_to_expiry"], r["strike"])),
        "meta": {
            "instruments_scanned": len(summaries),
            "used_bs_calculated_gamma": used_bs_count,
            "skipped_no_iv": skipped_no_iv,
            "risk_free_rate_used": GEX_RISK_FREE_RATE,
        },
    }


async def _gex_get(currency, force_refresh=False):
    currency = currency.upper()
    now = time.time()
    async with _gex_cache_lock:
        cached = _gex_cache.get(currency)
        if not force_refresh and cached and (now - cached["ts"]) < GEX_CACHE_TTL_SECONDS:
            return cached["data"]
    data = await _gex_compute_snapshot(currency)
    async with _gex_cache_lock:
        _gex_cache[currency] = {"data": data, "ts": now}
    return data


async def gex_background_loop():
    while True:
        for currency in GEX_SUPPORTED_CURRENCIES:
            try:
                await _gex_get(currency, force_refresh=True)
                print(f"[GEX] Refreshed {currency} snapshot.")
            except Exception as e:
                print(f"[GEX] Background refresh error ({currency}): {type(e).__name__}: {e}")
        await asyncio.sleep(GEX_CACHE_TTL_SECONDS)


@gex_router.get("/api/gex")
async def api_get_gex(currency: str = "BTC"):
    currency = currency.upper()
    if currency not in GEX_SUPPORTED_CURRENCIES:
        return {"error": f"Unsupported currency. Use one of: {GEX_SUPPORTED_CURRENCIES}"}
    return await _gex_get(currency)


@gex_router.post("/api/gex/refresh")
async def api_refresh_gex(currency: str = "BTC"):
    currency = currency.upper()
    if currency not in GEX_SUPPORTED_CURRENCIES:
        return {"error": f"Unsupported currency. Use one of: {GEX_SUPPORTED_CURRENCIES}"}
    data = await _gex_get(currency, force_refresh=True)
    return {
        "message": f"{currency} GEX refreshed",
        "condition": data.get("combined", {}).get("condition"),
        "near_term_condition": data.get("near_term", {}).get("condition"),
    }
# ============================================================================
# END GEX ENGINE
# ============================================================================

app = FastAPI(title="Order Flow X-Ray + Combined Signal Strategy + GEX")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(gex_router)

# ---------------- CONFIG ----------------
DEPTH_LEVELS = 20
DEFAULT_BALANCE = 100.0
DATA_FILE = "strategy_trading_data_v2.json"
FEE_PCT_PER_SIDE = 0.0001
MAX_HISTORY_CANDLES = 300
# -----------------------------------------

current_symbol = "btcusdt"
current_tick_size = 1.0

current_minute_ts = None
current_levels = defaultdict(lambda: {"bid": 0.0, "ask": 0.0})

balance = DEFAULT_BALANCE
positions = {}
trades_history = []
candle_history = []
trade_screenshots = []
factor_screenshots = []
SCREENSHOTS_DIR = "static/trade_screenshots"
FACTOR_SCREENSHOTS_DIR = "static/factor_screenshots"
MAX_SCREENSHOTS = 200
MAX_FACTOR_SCREENSHOTS = 200

strategy_paused = False

lock = asyncio.Lock()
factor_screenshots_lock = asyncio.Lock()

FACTOR_LIST = [
    "TICK_VOLUME", "CDD", "MEAN_REVERSION", "RSI",
    "Z_SCORE", "AGGRESSIVE_FLOW", "ICEBERG", "CVD_DIVERGENCE"
]

FACTOR_META = {
    "TICK_VOLUME": {"color": "#ed64a6", "label": "Tick Volume (ATS)"},
    "CDD": {"color": "#e8c547", "label": "Cumulative Delta Divergence"},
    "MEAN_REVERSION": {"color": "#dd6b20", "label": "Mean Reversion"},
    "RSI": {"color": "#fc8181", "label": "RSI"},
    "Z_SCORE": {"color": "#9f7aea", "label": "Z-Score"},
    "AGGRESSIVE_FLOW": {"color": "#f687b3", "label": "Aggressive Flow Ratio"},
    "ICEBERG": {"color": "#f56565", "label": "Iceberg Detection"},
    "CVD_DIVERGENCE": {"color": "#805ad5", "label": "CVD Divergence"}
}

factor_testing_14 = {
    factor: {
        "active": False, "current_direction": "NEUTRAL", "current_signal": "NEUTRAL",
        "entry_price": 0, "entry_time": 0, "win": 0, "loss": 0, "accuracy": 0, "total_trades": 0,
    } for factor in FACTOR_LIST
}

factor_testing_14_lock = asyncio.Lock()


def _ema(values, period):
    if not values or len(values) < period:
        return [values[-1]] if values else []
    k = 2 / (period + 1)
    ema = [values[0]]
    for v in values[1:]:
        ema.append(v * k + ema[-1] * (1 - k))
    return ema


def _calculate_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50
    changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [c if c > 0 else 0 for c in changes[-period:]]
    losses = [-c if c < 0 else 0 for c in changes[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _calculate_atr(candles, period=14):
    if len(candles) < period + 1:
        return 0
    trs = []
    for i in range(1, len(candles)):
        high = candles[i]["high"]
        low = candles[i]["low"]
        prev_close = candles[i-1]["close"]
        trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    if len(trs) < period:
        return 0
    return sum(trs[-period:]) / period


def get_14_factor_signals(candles):
    if not candles or len(candles) < 30:
        return {f: "NEUTRAL" for f in FACTOR_LIST}

    signals = {f: "NEUTRAL" for f in FACTOR_LIST}
    n = len(candles)
    current = candles[-1]
    prev20 = candles[max(0, n-21):n-1]
    prev10 = candles[max(0, n-11):n-1]
    vol = current.get("volume", 0)
    buy_vol = current.get("buy_vol", vol * 0.5)
    sell_vol = current.get("sell_vol", vol * 0.5)
    total_vol = buy_vol + sell_vol
    delta = current.get("delta", 0)
    current_range = current["high"] - current["low"]

    # TICK_VOLUME
    tick_vol = current.get("tick_volume", vol / 100)
    reg_vol = vol or 1
    ats = tick_vol / reg_vol
    ats_list = [c.get("volume", 0) / max(c.get("tick_volume", 1), 1) for c in candles[-100:]]
    ats_threshold = sorted(ats_list)[int(0.9 * len(ats_list))] if ats_list else 0
    ats_valid = ats > ats_threshold
    if ats_valid and delta > 0:
        signals["TICK_VOLUME"] = "BUY"
    elif ats_valid and delta < 0:
        signals["TICK_VOLUME"] = "SELL"
    else:
        signals["TICK_VOLUME"] = "NEUTRAL"

    # CDD
    prev20_high = max(c["high"] for c in prev20) if prev20 else current["high"]
    prev20_low = min(c["low"] for c in prev20) if prev20 else current["low"]
    prev20_delta_high = max(c.get("delta", 0) for c in prev20) if prev20 else 0
    prev20_delta_low = min(c.get("delta", 0) for c in prev20) if prev20 else 0
    if current["high"] > prev20_high and delta < prev20_delta_high:
        signals["CDD"] = "SELL"
    elif current["low"] < prev20_low and delta > prev20_delta_low:
        signals["CDD"] = "BUY"
    else:
        signals["CDD"] = "NEUTRAL"

    # MEAN_REVERSION
    closes = [c["close"] for c in candles]
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    if ema12 and ema26 and len(ema12) == len(ema26):
        macd_series = [ema12[i] - ema26[i] for i in range(len(ema12))]
    else:
        macd_series = [0.0]
    macd_line = macd_series[-1]
    if macd_line > 0:
        signals["MEAN_REVERSION"] = "SELL"
    elif macd_line < 0:
        signals["MEAN_REVERSION"] = "BUY"
    else:
        signals["MEAN_REVERSION"] = "NEUTRAL"

    # RSI
    rsi = _calculate_rsi(closes, 14)
    signals["RSI"] = "SELL" if rsi > 70 else ("BUY" if rsi < 30 else "NEUTRAL")

    # Z_SCORE
    mean = sum(closes[-20:]) / 20 if len(closes) >= 20 else closes[-1]
    variance = sum((c - mean) ** 2 for c in closes[-20:]) / 20 if len(closes) >= 20 else 0
    std = variance ** 0.5 if variance > 0 else 1
    z_score = (closes[-1] - mean) / std
    signals["Z_SCORE"] = "SELL" if z_score > 2 else ("BUY" if z_score < -2 else "NEUTRAL")

    # AGGRESSIVE_FLOW
    aggressive_sell = current.get("put_volume", sell_vol)
    aggressive_buy = current.get("call_volume", buy_vol)
    flow_ratio = aggressive_sell / aggressive_buy if aggressive_buy > 0 else 1
    signals["AGGRESSIVE_FLOW"] = "BUY" if flow_ratio > 1.2 else ("SELL" if flow_ratio < 0.8 else "NEUTRAL")

    # ICEBERG
    avg_volume_prev10 = sum(c.get("volume", 0) for c in prev10) / max(len(prev10), 1)
    vol_ratio = vol / (avg_volume_prev10 + 1e-9)
    avg_range_prev10 = sum((c["high"] - c["low"]) for c in prev10) / max(len(prev10), 1)
    range_ratio = current_range / (avg_range_prev10 + 1e-9)
    absorption_detected = (vol_ratio > 2.0) and (range_ratio < 0.6)
    if absorption_detected and sell_vol > buy_vol:
        signals["ICEBERG"] = "BUY"
    elif absorption_detected and buy_vol > sell_vol:
        signals["ICEBERG"] = "SELL"
    else:
        signals["ICEBERG"] = "NEUTRAL"

    # CVD_DIVERGENCE
    cvd_20_high = max(c.get("delta", 0) for c in prev20) if prev20 else 0
    cvd_20_low = min(c.get("delta", 0) for c in prev20) if prev20 else 0
    cvd_div_buy = current["close"] < prev20_low and delta > cvd_20_low
    cvd_div_sell = current["close"] > prev20_high and delta < cvd_20_high
    signals["CVD_DIVERGENCE"] = "BUY" if cvd_div_buy else ("SELL" if cvd_div_sell else "NEUTRAL")

    return signals


def _generate_factor_screenshot_sync(symbol, factor_name, direction, entry_price, exit_price, reason, entry_time, exit_time, out_path):
    if not SCREENSHOTS_AVAILABLE:
        return
    try:
        start_ts = entry_time - 100 * 60000
        end_ts = exit_time + 50 * 60000
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval=1m&startTime={start_ts}&endTime={end_ts}&limit=200"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return
        rows = resp.json()
        if not rows:
            return
        df = pd.DataFrame(rows, columns=[
            'time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('time', inplace=True)
        df = df[['open', 'high', 'low', 'close']].astype(float)
        mc = mpf.make_marketcolors(up="#3fdc9c", down="#ff6b6b", edge="inherit", wick="inherit", volume="inherit")
        style = mpf.make_mpf_style(
            base_mpl_style="dark_background", marketcolors=mc,
            facecolor="#161821", edgecolor="#262832", gridcolor="#1e2027", gridstyle="-",
            rc={"font.size": 9},
        )
        fig, ax = mpf.plot(df, type="candle", style=style, volume=False,
                            returnfig=True, figsize=(12, 6), tight_layout=True)
        ax = ax[0]
        ax.axhline(entry_price, color="#2b6cb0", linestyle="-", linewidth=1.5, label=f"Entry {entry_price}")
        exit_color = "#3fdc9c" if reason == "TARGET" else "#ff6b6b"
        exit_label = f"Target {exit_price}" if reason == "TARGET" else f"Stop {exit_price}"
        ax.axhline(exit_price, color=exit_color, linestyle="--", linewidth=1.5, label=exit_label)
        ax.legend(loc="upper left", fontsize=8, facecolor="#161821", edgecolor="#262832", labelcolor="#eaeaea")
        direction_text = "LONG" if direction == "BUY" else "SHORT"
        ax.set_title(f"{symbol} | {factor_name} | {direction_text} | {reason}", color="#eaeaea", fontsize=11)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=100, facecolor="#0e0f13")
        plt.close(fig)
    except Exception as e:
        print(f"Factor screenshot generation failed: {e}")


async def save_factor_screenshot(factor_name, direction, entry_price, exit_price, reason, entry_time, exit_time):
    if not SCREENSHOTS_AVAILABLE:
        return
    factor_dir = os.path.join(FACTOR_SCREENSHOTS_DIR, factor_name)
    os.makedirs(factor_dir, exist_ok=True)
    timestamp = int(time.time() * 1000)
    filename = f"{factor_name}_{entry_time}_{timestamp}.png"
    out_path = os.path.join(factor_dir, filename)
    await asyncio.to_thread(
        _generate_factor_screenshot_sync,
        current_symbol.upper(), factor_name, direction, entry_price, exit_price,
        reason, entry_time, exit_time, out_path
    )
    async with factor_screenshots_lock:
        factor_screenshots.insert(0, {
            "type": "factor", "factor": factor_name, "direction": direction,
            "entry_price": entry_price, "exit_price": exit_price, "reason": reason,
            "entry_time": entry_time, "exit_time": exit_time,
            "image_url": f"/static/factor_screenshots/{factor_name}/{filename}"
        })
        while len(factor_screenshots) > MAX_FACTOR_SCREENSHOTS:
            old = factor_screenshots.pop()
            old_path = old.get("image_url", "").lstrip("/")
            if old_path and os.path.exists(old_path):
                try:
                    os.remove(old_path)
                except Exception:
                    pass


async def update_14_factor_testing():
    global factor_testing_14
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={current_symbol.upper()}&interval=1m&limit=40"
        rows = await http_get_json(url, timeout=10)
        if not rows:
            return
        candles = []
        for k in rows:
            volume = float(k[5])
            buy_vol = float(k[9])
            sell_vol = max(volume - buy_vol, 0)
            candles.append({
                "open": float(k[1]), "high": float(k[2]), "low": float(k[3]), "close": float(k[4]),
                "volume": volume, "buy_vol": buy_vol, "sell_vol": sell_vol,
                "delta": buy_vol - sell_vol, "time": int(k[0])
            })
        current_price = candles[-1]["close"]
        signals = get_14_factor_signals(candles)
        async with factor_testing_14_lock:
            for factor in FACTOR_LIST:
                factor_data = factor_testing_14[factor]
                signal = signals.get(factor, "NEUTRAL")
                factor_data["current_signal"] = signal
                if factor_data["current_direction"] != signal and signal != "NEUTRAL":
                    if factor_data["active"]:
                        entry_price = factor_data["entry_price"]
                        entry_time = factor_data["entry_time"]
                        direction = factor_data["current_direction"]
                        exit_price = current_price
                        exit_time = int(time.time() * 1000)
                        reason = "TARGET"
                        if direction == "BUY" and signal == "SELL":
                            if current_price > entry_price:
                                factor_data["win"] += 1; reason = "TARGET"
                            else:
                                factor_data["loss"] += 1; reason = "STOP_LOSS"
                        elif direction == "SELL" and signal == "BUY":
                            if current_price < entry_price:
                                factor_data["win"] += 1; reason = "TARGET"
                            else:
                                factor_data["loss"] += 1; reason = "STOP_LOSS"
                        factor_data["active"] = False
                        factor_data["total_trades"] += 1
                        asyncio.create_task(save_factor_screenshot(
                            factor, direction, entry_price, exit_price, reason, entry_time, exit_time
                        ))
                    if signal != "NEUTRAL":
                        factor_data["active"] = True
                        factor_data["entry_price"] = current_price
                        factor_data["entry_time"] = int(time.time() * 1000)
                    factor_data["current_direction"] = signal
                    total = factor_data["win"] + factor_data["loss"]
                    factor_data["accuracy"] = round((factor_data["win"] / total) * 100, 2) if total > 0 else 0
                if signal == "NEUTRAL":
                    factor_data["current_direction"] = factor_data["current_direction"] or "NEUTRAL"
    except Exception as e:
        print(f"Factor testing error: {e}")


async def factor_testing_14_loop():
    while True:
        await asyncio.sleep(1)
        await update_14_factor_testing()


def determine_tick_size(symbol):
    try:
        resp = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.upper()}", timeout=5)
        price = float(resp.json()["price"])
    except Exception:
        price = 100.0
    if price > 10000: return 5.0
    elif price > 1000: return 1.0
    elif price > 100: return 0.1
    elif price > 10: return 0.01
    elif price > 1: return 0.001
    else: return 0.0001


def get_timeframe_seconds(timeframe):
    tf_map = {"1m": 60, "5m": 300, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}
    return tf_map.get(timeframe, 60)


def load_trading_data():
    global balance, positions, trades_history, trade_screenshots
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            balance = saved.get("balance", DEFAULT_BALANCE)
            saved_positions = saved.get("positions")
            if saved_positions is None:
                positions = {}
            else:
                positions = saved_positions
            trades_history = saved.get("trades_history", [])
            trade_screenshots = saved.get("trade_screenshots", [])
            print(f"Loaded saved trading data: balance={balance}, positions={len(positions)}, trades={len(trades_history)}")
        except Exception as e:
            print("Could not load trading data file:", e)


def save_trading_data():
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "balance": balance, "positions": positions,
                "trades_history": trades_history, "trade_screenshots": trade_screenshots,
            }, f, indent=2)
    except Exception as e:
        print("Could not save trading data file:", e)


def bucket_price(price):
    return round(math.floor(price / current_tick_size) * current_tick_size, 8)


def floor_qty(qty):
    return int(math.floor(qty))


def classify(bid_total, ask_total):
    if bid_total == 0 and ask_total == 0:
        return "WAIT", "Buyers and sellers are mixed, no trade recommended.", "Neutral"
    ratio_bid = (bid_total / ask_total) if ask_total > 0 else float("inf")
    ratio_ask = (ask_total / bid_total) if bid_total > 0 else float("inf")
    if ratio_bid >= 2:
        state = "LONG"
        if ratio_bid < 5:
            box2 = "Buyers are aggressive but strength is medium."; box3 = "Bullish"
        elif ratio_bid < 10:
            box2 = "Market is under buyers' control."; box3 = "Strong Bullish"
        else:
            box2 = "Market is under buyers' control."; box3 = "Very Strong Bullish"
        return state, box2, box3
    if ratio_ask >= 2:
        state = "SHORT"
        if ratio_ask < 5:
            box2 = "Sellers are aggressive but strength is medium."; box3 = "Bearish"
        elif ratio_ask < 10:
            box2 = "Market is under sellers' control."; box3 = "Strong Bearish"
        else:
            box2 = "Market is under sellers' control."; box3 = "Very Strong Bearish"
        return state, box2, box3
    return "WAIT", "Buyers and sellers are mixed, no trade recommended.", "Neutral"


def classify_time_xray_strength(ask_volume, bid_volume):
    if ask_volume == 0 and bid_volume == 0: return "Neutral"
    if bid_volume == 0:
        return "Very Strong Bearish" if ask_volume > 0 else "Neutral"
    if ask_volume == 0:
        return "Very Strong Bullish" if bid_volume > 0 else "Neutral"
    if bid_volume > ask_volume:
        bid_ratio = bid_volume / ask_volume
        if 1 <= bid_ratio < 2: return "Neutral"
        elif 2 <= bid_ratio < 5: return "Bullish"
        elif 5 <= bid_ratio < 10: return "Strong Bullish"
        else: return "Very Strong Bullish"
    elif ask_volume > bid_volume:
        ask_ratio = ask_volume / bid_volume
        if 1 <= ask_ratio < 2: return "Neutral"
        elif 2 <= ask_ratio < 5: return "Bearish"
        elif 5 <= ask_ratio < 10: return "Strong Bearish"
        else: return "Very Strong Bearish"
    return "Neutral"


def compute_metrics():
    if current_minute_ts is None:
        return {
            "time": None, "levels": [], "bid_total": 0, "ask_total": 0,
            "delta": 0, "volume": 0, "state": "WAIT", "box2": "", "box3": "Neutral",
            "mid_price": None, "best_bid": None, "best_ask": None,
        }
    levels_list = [
        {"price": price, "bid": floor_qty(v["bid"]), "ask": floor_qty(v["ask"])}
        for price, v in sorted(current_levels.items(), key=lambda x: -x[0])
        if v["bid"] > 0 or v["ask"] > 0
    ]
    bid_total = sum(lvl["bid"] for lvl in levels_list)
    ask_total = sum(lvl["ask"] for lvl in levels_list)
    delta = bid_total - ask_total
    volume = bid_total + ask_total
    state, box2, box3 = classify(bid_total, ask_total)
    bid_prices = [p for p, v in current_levels.items() if v["bid"] > 0]
    ask_prices = [p for p, v in current_levels.items() if v["ask"] > 0]
    best_bid = max(bid_prices) if bid_prices else None
    best_ask = min(ask_prices) if ask_prices else None
    if best_bid is not None and best_ask is not None:
        mid_price = (best_bid + best_ask) / 2
    elif best_bid is not None:
        mid_price = best_bid
    elif best_ask is not None:
        mid_price = best_ask
    else:
        mid_price = None
    return {
        "time": current_minute_ts, "levels": levels_list,
        "bid_total": bid_total, "ask_total": ask_total,
        "delta": delta, "volume": volume,
        "state": state, "box2": box2, "box3": box3,
        "mid_price": mid_price, "best_bid": best_bid, "best_ask": best_ask,
    }


async def depth_listener():
    global current_minute_ts, current_levels

    print("[Depth] Using REST polling (every 1s) via Binance /api/v3/depth")

    while True:
        try:
            url = (f"https://api.binance.com/api/v3/depth?"
                   f"symbol={current_symbol.upper()}&limit={DEPTH_LEVELS}")

            data = await http_get_json(url, timeout=8)
            if data is None:
                await asyncio.sleep(2)
                continue

            now_ms = int(time.time() * 1000)
            minute_ts = (now_ms // 60000) * 60000

            async with lock:
                if current_minute_ts is None:
                    current_minute_ts = minute_ts

                if minute_ts != current_minute_ts:
                    old_metrics = compute_metrics()
                    candle_history.append({
                        "time": old_metrics["time"], "delta": old_metrics["delta"],
                        "volume": old_metrics["volume"], "state": old_metrics["state"],
                        "box3": old_metrics["box3"],
                    })
                    if len(candle_history) > MAX_HISTORY_CANDLES:
                        candle_history.pop(0)
                    current_minute_ts = minute_ts
                    current_levels = defaultdict(lambda: {"bid": 0.0, "ask": 0.0})

                for price_str, qty_str in data.get("bids", []):
                    price = bucket_price(float(price_str))
                    qty = float(qty_str)
                    if qty > current_levels[price]["bid"]:
                        current_levels[price]["bid"] = qty

                for price_str, qty_str in data.get("asks", []):
                    price = bucket_price(float(price_str))
                    qty = float(qty_str)
                    if qty > current_levels[price]["ask"]:
                        current_levels[price]["ask"] = qty

        except Exception as e:
            print(f"[Depth] REST error: {type(e).__name__}: {e}")

        await asyncio.sleep(1)


# ============================================================================
# SPOOFING DETECTION (rewritten - see module docstring for the fixes)
# ============================================================================
_spoofing_lock = asyncio.Lock()
_spoofing_local_book = {"bid": {}, "ask": {}}
_spoofing_tracked_orders = {}
_spoofing_recent_trades = deque(maxlen=5000)   # (timestamp, price, qty) from @aggTrade
spoofing_events = []
SPOOFING_MAX_EVENTS = 50

SPOOFING_MIN_NOTIONAL_USD = 200_000.0
SPOOFING_MAX_TIME = 8.0
SPOOFING_MIN_SCORE = 40

# *** FIX: was 0.0005 (0.05%) -> ~$45 on BTC, so trades happened inside that
# band constantly and blocked almost every detection. Now a much tighter
# band, and paired with a VOLUME check (see SPOOFING_FILL_VOLUME_FRACTION)
# instead of a plain "any trade nearby = discard" boolean. ***
SPOOFING_PRICE_TOLERANCE_PCT = 0.00005   # ~$4.50 on a $90k BTC price
SPOOFING_FILL_VOLUME_FRACTION = 0.3      # if >=30% of the vanished size actually traded nearby, treat as a real fill

_spoofing_running = True
_spoofing_last_applied_u = None


async def _spoofing_fetch_snapshot(symbol):
    url = f"https://fapi.binance.com/fapi/v1/depth?symbol={symbol.upper()}&limit=1000"
    return await http_get_json(url, timeout=10)


async def _spoofing_apply_event(data):
    now = time.time()
    for price_str, qty_str in data.get("b", []):
        await _spoofing_process_level("bid", float(price_str), float(qty_str), now)
    for price_str, qty_str in data.get("a", []):
        await _spoofing_process_level("ask", float(price_str), float(qty_str), now)


async def _spoofing_process_level(side, price, new_qty, now):
    """
    *** FIX: removed the "partially_filled -> disqualify forever" flag. ***
    A depth level is the SUM of every trader's resting size at that price,
    not one order - some unrelated trader nudging their own order at the
    same price used to permanently disqualify a genuine large spoof sitting
    there too. Now we just track the max size seen, and when the level fully
    empties, check the trade tape (see _spoofing_check_and_emit) to see if
    that emptying was actually explained by real trading volume.
    """
    prev_qty = _spoofing_local_book[side].get(price, 0)
    key = f"{side}_{price}"
    notional = price * new_qty

    if prev_qty == 0 and notional >= SPOOFING_MIN_NOTIONAL_USD:
        async with _spoofing_lock:
            _spoofing_tracked_orders[key] = {
                "side": side,
                "price": price,
                "first_seen": now,
                "max_qty": new_qty,
            }
    else:
        tracked = _spoofing_tracked_orders.get(key)
        if tracked:
            if new_qty > tracked["max_qty"]:
                tracked["max_qty"] = new_qty
            if new_qty == 0 and (tracked["price"] * tracked["max_qty"]) >= SPOOFING_MIN_NOTIONAL_USD:
                await _spoofing_check_and_emit(tracked, now)
                async with _spoofing_lock:
                    _spoofing_tracked_orders.pop(key, None)

    if new_qty == 0:
        _spoofing_local_book[side].pop(price, None)
    else:
        _spoofing_local_book[side][price] = new_qty


def _spoofing_traded_volume_near(price, start_time, end_time):
    """
    *** FIX: volume-based cross-check instead of a boolean "any trade nearby". ***
    Sums the actual traded quantity at (or very near) this price during the
    window the order was resting. Only a MEANINGFUL amount of real trading
    there (see SPOOFING_FILL_VOLUME_FRACTION) counts as a genuine fill/sweep;
    a stray unrelated trade printing nearby no longer disqualifies the whole
    candidate.
    """
    tolerance = price * SPOOFING_PRICE_TOLERANCE_PCT
    total = 0.0
    for t, p, q in _spoofing_recent_trades:
        if start_time <= t <= end_time and abs(p - price) <= tolerance:
            total += q
    return total


async def _spoofing_check_and_emit(tracked, now):
    time_in_book = now - tracked["first_seen"]
    notional = tracked["price"] * tracked["max_qty"]

    if time_in_book > SPOOFING_MAX_TIME:
        return
    if notional < SPOOFING_MIN_NOTIONAL_USD:
        return

    traded_volume = _spoofing_traded_volume_near(tracked["price"], tracked["first_seen"], now)
    if traded_volume >= tracked["max_qty"] * SPOOFING_FILL_VOLUME_FRACTION:
        return  # meaningful real trading happened here - a fill/sweep, not a spoof

    score = _spoofing_calculate_score(notional, time_in_book)
    if score < SPOOFING_MIN_SCORE:
        return

    event = {
        "symbol": current_symbol.upper(),
        "timestamp": int(now * 1000),
        "price": round(tracked["price"], 2),
        "size": round(tracked["max_qty"], 4),
        "notional_usd": round(notional, 2),
        "side": "SELL" if tracked["side"] == "ask" else "BUY",
        "time_in_book": round(time_in_book, 2),
        "verified_no_fill": True,
        "nearby_traded_volume": round(traded_volume, 4),
        "score": score,
    }

    async with _spoofing_lock:
        spoofing_events.insert(0, event)
        while len(spoofing_events) > SPOOFING_MAX_EVENTS:
            spoofing_events.pop()


def _spoofing_calculate_score(notional_usd, time_in_book):
    score = 40

    if notional_usd >= 5_000_000: score += 20
    elif notional_usd >= 2_000_000: score += 16
    elif notional_usd >= 1_000_000: score += 12
    elif notional_usd >= 750_000: score += 8
    elif notional_usd >= 500_000: score += 4

    if time_in_book < 0.2: score += 30
    elif time_in_book < 0.5: score += 25
    elif time_in_book < 1.0: score += 20
    elif time_in_book < 2.0: score += 15
    elif time_in_book < 5.0: score += 10

    return min(score, 100)


async def _spoofing_cleanup(now):
    async with _spoofing_lock:
        stale = [k for k, t in _spoofing_tracked_orders.items()
                 if now - t["first_seen"] > SPOOFING_MAX_TIME * 2]
        for k in stale:
            _spoofing_tracked_orders.pop(k, None)


async def _spoofing_ws_loop():
    global _spoofing_local_book, _spoofing_running, _spoofing_last_applied_u

    while _spoofing_running:
        sym = current_symbol.lower()
        stream_url = f"wss://fstream.binance.com/stream?streams={sym}@depth@100ms/{sym}@aggTrade"
        print(f"[Spoofing] Connecting to {stream_url}")

        buffered_events = []
        synced = False
        _spoofing_local_book = {"bid": {}, "ask": {}}
        _spoofing_tracked_orders.clear()
        _spoofing_recent_trades.clear()
        _spoofing_last_applied_u = None

        try:
            async with websockets.connect(stream_url, ping_interval=20, ping_timeout=10) as ws:
                print(f"[Spoofing] WebSocket connected for {sym}")

                while _spoofing_running:
                    if current_symbol.lower() != sym:
                        print("[Spoofing] Symbol changed, reconnecting...")
                        break

                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                    except asyncio.TimeoutError:
                        continue

                    payload = json.loads(msg)
                    stream_name = payload.get("stream", "")
                    data = payload.get("data", {})

                    if stream_name.endswith("@aggTrade"):
                        try:
                            _spoofing_recent_trades.append(
                                (time.time(), float(data["p"]), float(data["q"]))
                            )
                        except Exception:
                            pass
                        continue

                    if data.get("e") != "depthUpdate":
                        continue

                    if not synced:
                        buffered_events.append(data)
                        if len(buffered_events) == 1:
                            snapshot = await _spoofing_fetch_snapshot(sym)
                            if snapshot is None or "lastUpdateId" not in snapshot:
                                await asyncio.sleep(2)
                                buffered_events = []
                                continue
                            last_update_id = snapshot["lastUpdateId"]
                            buffered_events = [e for e in buffered_events if e.get("u", 0) > last_update_id]
                            first_valid = None
                            for e in buffered_events:
                                if e.get("U", 0) <= last_update_id + 1 <= e.get("u", 0):
                                    first_valid = e
                                    break
                            if first_valid is None:
                                continue
                            async with _spoofing_lock:
                                _spoofing_local_book["bid"] = {
                                    float(p): float(q) for p, q in snapshot.get("bids", [])
                                }
                                _spoofing_local_book["ask"] = {
                                    float(p): float(q) for p, q in snapshot.get("asks", [])
                                }
                            idx = buffered_events.index(first_valid)
                            for e in buffered_events[idx:]:
                                await _spoofing_apply_event(e)
                            _spoofing_last_applied_u = buffered_events[-1]["u"]
                            synced = True
                            buffered_events = []
                            print(f"[Spoofing] Synced order book for {sym} (lastUpdateId={last_update_id})")
                            print(f"[Spoofing] Tracking: threshold=${SPOOFING_MIN_NOTIONAL_USD:,.0f}, "
                                  f"max_time={SPOOFING_MAX_TIME}s, tolerance={SPOOFING_PRICE_TOLERANCE_PCT*100:.4f}%")
                        continue

                    if data.get("pu") != _spoofing_last_applied_u:
                        print("[Spoofing] Sequence gap detected, resyncing...")
                        break

                    await _spoofing_apply_event(data)
                    _spoofing_last_applied_u = data["u"]
                    await _spoofing_cleanup(time.time())

        except Exception as e:
            print(f"[Spoofing] WS error: {type(e).__name__}: {e}")
            await asyncio.sleep(5)


# ============================================================================
# API ENDPOINTS
# ============================================================================

class SymbolUpdate(BaseModel):
    symbol: str


@app.post("/api/set_symbol")
async def set_symbol(update: SymbolUpdate):
    global current_symbol, current_tick_size
    new_symbol = update.symbol.strip().lower()
    if not new_symbol:
        return {"error": "Symbol cannot be empty"}
    new_tick_size = await asyncio.to_thread(determine_tick_size, new_symbol)
    async with lock:
        current_symbol = new_symbol
        current_tick_size = new_tick_size
    return {"symbol": current_symbol.upper(), "tick_size": current_tick_size}


@app.get("/api/symbol")
async def get_symbol():
    return {"symbol": current_symbol.upper(), "tick_size": current_tick_size}


@app.get("/api/candle")
async def get_candle():
    async with lock:
        m = compute_metrics()
    return {
        "symbol": current_symbol.upper(), "tick_size": current_tick_size,
        "time": m["time"], "levels": m["levels"],
        "bid_total": m["bid_total"], "ask_total": m["ask_total"],
        "delta": m["delta"], "volume": m["volume"],
        "state": m["state"], "box2": m["box2"], "box3": m["box3"],
        "best_bid": m["best_bid"], "best_ask": m["best_ask"],
    }


# ============================================================================
# MARKET METRICS — parallelized + cached (FIX)
# ============================================================================
_metrics_cache = {"data": None, "ts": 0}
_METRICS_CACHE_TTL = 60


@app.get("/api/market_metrics")
async def get_market_metrics():
    global _metrics_cache
    sym = current_symbol.upper()
    now = time.time()

    # *** FIX: fetch these three concurrently instead of one-by-one, and via
    # the shared cache so liquidity_magnet/zones calls don't duplicate work. ***
    premium_task = cached_http_get_json(
        f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={sym}", ttl=4, timeout=8)
    oi_task = cached_http_get_json(
        f"https://fapi.binance.com/fapi/v1/openInterest?symbol={sym}", ttl=4, timeout=8)
    ticker_task = cached_http_get_json(
        f"https://fapi.binance.com/fapi/v1/ticker/24hr?symbol={sym}", ttl=4, timeout=10)

    premium, oi_data, ticker = await asyncio.gather(premium_task, oi_task, ticker_task)

    price = 0.0
    oi_btc = 0.0
    funding_rate = 0.0
    volume_usd = 0.0
    high_24 = 0.0
    low_24 = 0.0
    mark_price = 0.0

    if premium:
        try:
            if "lastFundingRate" in premium:
                funding_rate = float(premium["lastFundingRate"]) * 100
            if "markPrice" in premium:
                mark_price = float(premium["markPrice"])
        except Exception:
            pass

    if oi_data and "openInterest" in oi_data:
        try:
            oi_btc = float(oi_data["openInterest"])
        except Exception:
            pass

    if ticker and "lastPrice" in ticker:
        try:
            price = float(ticker["lastPrice"])
            volume_usd = float(ticker.get("quoteVolume", 0))
            high_24 = float(ticker.get("highPrice", 0))
            low_24 = float(ticker.get("lowPrice", 0))
        except Exception:
            pass

    if price <= 0 and mark_price > 0:
        price = mark_price
        print(f"[Metrics] Using markPrice fallback: {mark_price}")

    if price <= 0:
        tp = await cached_http_get_json(
            f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={sym}", ttl=2, timeout=5)
        if tp and "price" in tp:
            try:
                price = float(tp["price"])
                print(f"[Metrics] Using ticker/price fallback: {price}")
            except Exception:
                pass

    if high_24 <= 0 or low_24 <= 0 or volume_usd <= 0:
        klines = await cached_http_get_json(
            f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval=1h&limit=24", ttl=30, timeout=8)
        if klines and len(klines) > 0:
            try:
                highs = [float(k[2]) for k in klines]
                lows = [float(k[3]) for k in klines]
                quote_vols = [float(k[7]) for k in klines]
                if high_24 <= 0: high_24 = max(highs)
                if low_24 <= 0: low_24 = min(lows)
                if volume_usd <= 0: volume_usd = sum(quote_vols)
                if price <= 0: price = float(klines[-1][4])
            except Exception:
                pass

    oi_usd = oi_btc * price if price > 0 else 0.0

    if price <= 0 and oi_btc <= 0:
        cached = _metrics_cache.get("data")
        if cached and (now - _metrics_cache["ts"]) < _METRICS_CACHE_TTL * 5:
            print(f"[Metrics] All endpoints failed — returning cached data (age {(now - _metrics_cache['ts']):.0f}s)")
            cached_copy = dict(cached)
            cached_copy["cached"] = True
            return cached_copy

    result = {
        "symbol": sym,
        "timestamp": int(time.time() * 1000),
        "price": price,
        "open_interest_btc": round(oi_btc, 4),
        "open_interest_usd": round(oi_usd, 2),
        "funding_rate_pct": round(funding_rate, 6),
        "volume_24h_usd": round(volume_usd, 2),
        "high_24h": high_24,
        "low_24h": low_24,
        "cached": False,
    }

    if price > 0 and oi_btc > 0:
        _metrics_cache["data"] = result
        _metrics_cache["ts"] = now

    return result


@app.get("/api/liquidity_magnet")
async def get_liquidity_magnet():
    try:
        sym = current_symbol.upper()

        # *** FIX: cached depth fetch, shared with /api/liquidity_zones ***
        depth = await cached_http_get_json(
            f"https://fapi.binance.com/fapi/v1/depth?symbol={sym}&limit=1000", ttl=3, timeout=8)
        if not depth:
            return {"error": "Could not fetch order book"}

        ticker = await cached_http_get_json(
            f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={sym}", ttl=2, timeout=5)
        current_price = float(ticker["price"]) if ticker else 0.0
        if current_price <= 0:
            return {"error": "Could not fetch price"}

        all_clusters = []

        for price_str, qty_str in depth.get("bids", []):
            price = float(price_str)
            qty = float(qty_str)
            notional = price * qty
            all_clusters.append({
                "price": price, "qty": qty, "size_usd": notional, "type": "BUY"
            })

        for price_str, qty_str in depth.get("asks", []):
            price = float(price_str)
            qty = float(qty_str)
            notional = price * qty
            all_clusters.append({
                "price": price, "qty": qty, "size_usd": notional, "type": "SELL"
            })

        # *** CHANGED: threshold is now $10M (was $5M) ***
        MIN_CLUSTER_SIZE = 10_000_000
        big_clusters = [c for c in all_clusters if c["size_usd"] >= MIN_CLUSTER_SIZE]

        if not big_clusters:
            # Fallback: if no $10M+ cluster exists, show top 10 largest instead
            all_clusters.sort(key=lambda x: x["size_usd"], reverse=True)
            big_clusters = all_clusters[:10]

        for c in big_clusters:
            signed_dist = (c["price"] - current_price) / current_price * 100
            c["distance_pct"] = round(signed_dist, 2)
            c["abs_distance_pct"] = abs(signed_dist)
            c["score"] = c["size_usd"] / (1 + abs(signed_dist) / 100)

        big_clusters.sort(key=lambda x: x["score"], reverse=True)
        magnet = big_clusters[0]

        return {
            "symbol": sym,
            "timestamp": int(time.time() * 1000),
            "current_price": round(current_price, 2),
            "magnet_price": round(magnet["price"], 2),
            "magnet_qty": round(magnet["qty"], 4),
            "magnet_size_usd": round(magnet["size_usd"], 2),
            "magnet_type": "BUY WALL" if magnet["type"] == "BUY" else "SELL WALL",
            "distance_pct": magnet["distance_pct"],
            "top_clusters": [
                {
                    "price": round(c["price"], 2),
                    "size_usd": round(c["size_usd"], 2),
                    "type": c["type"],
                    "distance_pct": c["distance_pct"],
                }
                for c in big_clusters[:8]
            ],
        }
    except Exception as e:
        print(f"[Liquidity Magnet] Error: {type(e).__name__}: {e}")
        return {"error": str(e)}


# ============================================================================
# LIQUIDITY TARGET ZONES
# ============================================================================
@app.get("/api/liquidity_zones")
async def get_liquidity_zones():
    try:
        sym = current_symbol.upper()

        # *** FIX: reuses the SAME cached depth fetch as liquidity_magnet - ***
        depth = await cached_http_get_json(
            f"https://fapi.binance.com/fapi/v1/depth?symbol={sym}&limit=1000", ttl=3, timeout=8)
        if not depth:
            return {"error": "Could not fetch order book"}

        ticker = await cached_http_get_json(
            f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={sym}", ttl=2, timeout=5)
        current_price = float(ticker["price"]) if ticker else 0.0
        if current_price <= 0:
            return {"error": "Could not fetch price"}

        all_levels = []

        for price_str, qty_str in depth.get("bids", []):
            price = float(price_str)
            qty = float(qty_str)
            notional = price * qty
            all_levels.append({
                "price": price, "qty": qty, "notional": notional, "type": "BUY"
            })

        for price_str, qty_str in depth.get("asks", []):
            price = float(price_str)
            qty = float(qty_str)
            notional = price * qty
            all_levels.append({
                "price": price, "qty": qty, "notional": notional, "type": "SELL"
            })

        # *** CHANGED: threshold is now $10M (was $5M) ***
        MIN_WALL_USD = 10_000_000
        walls = [lvl for lvl in all_levels if lvl["notional"] >= MIN_WALL_USD]

        if not walls:
            # Fallback: if no $10M+ wall exists, show top 10 largest instead
            all_levels.sort(key=lambda x: x["notional"], reverse=True)
            walls = all_levels[:10]

        walls.sort(key=lambda x: x["notional"], reverse=True)
        top = walls[:10]

        for w in top:
            n = w["notional"]
            # *** CHANGED: score tiers (10M minimum) ***
            #   $50M+  → 99  (Giant wall)
            #   $25M+  → 85  (Very strong)
            #   $15M+  → 70  (Strong)
            #   $10M+  → 55  (Normal)
            if n >= 50_000_000:
                w["score"] = 99
            elif n >= 25_000_000:
                w["score"] = 85
            elif n >= 15_000_000:
                w["score"] = 70
            elif n >= 10_000_000:
                w["score"] = 55
            else:
                w["score"] = 40
            w["distance_pct"] = round((w["price"] - current_price) / current_price * 100, 2)

        return {
            "symbol": sym,
            "timestamp": int(time.time() * 1000),
            "current_price": round(current_price, 2),
            "zones": [
                {
                    "price": round(w["price"], 2),
                    "notional_usd": round(w["notional"], 2),
                    "type": "Buy Wall" if w["type"] == "BUY" else "Sell Wall",
                    "distance_pct": w["distance_pct"],
                    "score": w["score"],
                }
                for w in top
            ],
        }
    except Exception as e:
        print(f"[Liquidity Zones] Error: {type(e).__name__}: {e}")
        return {"error": str(e)}


@app.get("/api/spoofing")
async def get_spoofing_events():
    sym = current_symbol.upper()
    async with _spoofing_lock:
        events_copy = [dict(e) for e in spoofing_events if e.get("symbol") == sym][:20]
        tracked_count = len(_spoofing_tracked_orders)
        all_events_count = len(spoofing_events)
        recent_trades_count = len(_spoofing_recent_trades)

    return {
        "symbol": sym,
        "timestamp": int(time.time() * 1000),
        "events": events_copy,
        "total_tracked": tracked_count,
        "diagnostics": {
            "total_events_all_symbols": all_events_count,
            "recent_trades_buffered": recent_trades_count,
            "ws_running": _spoofing_running,
            "ws_synced": _spoofing_last_applied_u is not None,
            "min_notional_usd": SPOOFING_MIN_NOTIONAL_USD,
            "max_time_sec": SPOOFING_MAX_TIME,
        }
    }


@app.get("/api/cvd")
async def get_cvd():
    try:
        sym = current_symbol.upper()
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval=15m&limit=96"
        rows = await cached_http_get_json(url, ttl=10, timeout=8)

        if not rows:
            return {"error": "Could not fetch CVD data"}

        candles = []
        total_buy = 0.0
        total_sell = 0.0

        for k in rows:
            vol = float(k[5])
            buy = float(k[9])
            sell = vol - buy
            delta = buy - sell

            total_buy += buy
            total_sell += sell

            candles.append({
                "time": int(k[0]),
                "delta": round(delta, 4),
                "buy": round(buy, 4),
                "sell": round(sell, 4),
            })

        total_delta = total_buy - total_sell

        recent = sum(c["delta"] for c in candles[-8:])
        if recent > 0:
            trend = "Bull"
        elif recent < 0:
            trend = "Bear"
        else:
            trend = "Neutral"

        return {
            "symbol": sym,
            "timestamp": int(time.time() * 1000),
            "candles": candles,
            "total_buy": round(total_buy, 2),
            "total_sell": round(total_sell, 2),
            "total_delta": round(total_delta, 2),
            "trend": trend,
            "period": "24H (15m)",
        }
    except Exception as e:
        print(f"[CVD] Error: {type(e).__name__}: {e}")
        return {"error": str(e)}


@app.get("/api/time_xray")
async def get_time_xray(timeframe: str = "1m"):
    valid_timeframes = ["1m", "5m", "30m", "1h", "4h", "1d"]
    if timeframe not in valid_timeframes:
        timeframe = "1m"
    try:
        tf_seconds = get_timeframe_seconds(timeframe)
        now_ms = int(time.time() * 1000)
        candle_start_ms = (now_ms // (tf_seconds * 1000)) * (tf_seconds * 1000)
        bid_levels = defaultdict(float)
        ask_levels = defaultdict(float)
        start_time = candle_start_ms
        end_time = now_ms
        tick_size = current_tick_size
        if tick_size <= 0:
            tick_size = determine_tick_size(current_symbol)
        max_pages = 50
        page_count = 0
        while page_count < max_pages:
            url = f"https://api.binance.com/api/v3/aggTrades?symbol={current_symbol.upper()}&startTime={start_time}&endTime={end_time}&limit=1000"
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200: break
            trades = resp.json()
            if not trades: break
            for trade in trades:
                price = float(trade["p"]); qty = float(trade["q"])
                bucket_p = round(math.floor(price / tick_size) * tick_size, 8)
                if trade["m"]: ask_levels[bucket_p] += qty
                else: bid_levels[bucket_p] += qty
            if len(trades) < 1000: break
            start_time = trades[-1]["T"] + 1
            page_count += 1
            if start_time >= end_time: break
        levels_list = []
        all_prices = sorted(set(list(bid_levels.keys()) + list(ask_levels.keys())), reverse=True)
        for price in all_prices[:50]:
            levels_list.append({"price": price, "bid": floor_qty(bid_levels.get(price, 0)), "ask": floor_qty(ask_levels.get(price, 0))})
        total_bid_volume = int(sum(bid_levels.values()))
        total_ask_volume = int(sum(ask_levels.values()))
        strength = classify_time_xray_strength(total_ask_volume, total_bid_volume)
        klines_url = f"https://api.binance.com/api/v3/klines?symbol={current_symbol.upper()}&interval={timeframe}&limit=100"
        klines_resp = requests.get(klines_url, timeout=10)
        candles = []
        binance_volume = binance_buy_volume = binance_sell_volume = 0
        if klines_resp.status_code == 200:
            for k in klines_resp.json():
                candles.append({
                    "time": int(k[0]), "open": float(k[1]), "high": float(k[2]),
                    "low": float(k[3]), "close": float(k[4]), "volume": float(k[5]),
                    "close_time": int(k[6]), "quote_volume": float(k[7]),
                    "trades": int(k[8]), "buy_volume": float(k[9]),
                    "sell_volume": max(float(k[5]) - float(k[9]), 0),
                })
            if candles:
                cc = candles[-1]
                binance_volume = cc["volume"]; binance_buy_volume = cc["buy_volume"]; binance_sell_volume = cc["sell_volume"]
        if total_bid_volume == 0 and total_ask_volume == 0:
            total_bid_volume = int(binance_buy_volume); total_ask_volume = int(binance_sell_volume)
        live_price = None
        ticker_resp = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={current_symbol.upper()}", timeout=5)
        if ticker_resp.status_code == 200:
            live_price = float(ticker_resp.json()["price"])
        ratio_value = round(total_ask_volume / total_bid_volume, 4) if total_bid_volume > 0 else (999.99 if total_ask_volume > 0 else 0)
        return {
            "symbol": current_symbol.upper(), "timeframe": timeframe,
            "candle_start_time": candle_start_ms, "levels": levels_list[:50],
            "bid_volume": total_bid_volume, "ask_volume": total_ask_volume,
            "total_volume": total_bid_volume + total_ask_volume,
            "binance_volume": binance_volume, "binance_buy_volume": binance_buy_volume,
            "binance_sell_volume": binance_sell_volume,
            "strength": strength, "ratio": ratio_value,
            "candles": candles, "live_price": live_price,
            "timestamp": int(time.time() * 1000),
        }
    except Exception as e:
        print(f"Time X-Ray Error: {e}")
        return {"error": str(e)}


@app.get("/api/history")
async def get_history():
    async with lock:
        m = compute_metrics()
        current = None
        if m["time"] is not None:
            current = {"time": m["time"], "delta": m["delta"], "volume": m["volume"], "state": m["state"], "box3": m["box3"]}
        return {"history": list(candle_history), "current": current}


def compute_direction_score(symbol, interval="1m"):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit=40"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return {"sell_score": 0, "buy_score": 0, "error": "Binance API failed"}
        rows = resp.json()
        if not rows or len(rows) < 35:
            return {"sell_score": 0, "buy_score": 0, "error": "Not enough data"}
        def compute_delta(k):
            volume = float(k[5]); buy_vol = float(k[9])
            sell_vol = max(volume - buy_vol, 0)
            return {"open": float(k[1]), "high": float(k[2]), "low": float(k[3]), "close": float(k[4]),
                    "volume": volume, "buy_vol": buy_vol, "sell_vol": sell_vol, "delta": buy_vol - sell_vol}
        candles = [compute_delta(k) for k in rows]
        n = len(candles)
        current = candles[n - 1]
        prev20 = candles[n - 21:n - 1]
        prev10 = candles[n - 11:n - 1]
        D = "DOWN" if current["delta"] < 0 else "UP"
        prev20_high = max(c["high"] for c in prev20)
        prev20_low = min(c["low"] for c in prev20)
        prev20_delta_high = max(c["delta"] for c in prev20)
        prev20_delta_low = min(c["delta"] for c in prev20)
        CDD = "NEUTRAL"
        if current["high"] > prev20_high and current["delta"] < prev20_delta_high: CDD = "DOWN"
        elif current["low"] < prev20_low and current["delta"] > prev20_delta_low: CDD = "UP"
        v_avg = sum(c["volume"] for c in prev10) / len(prev10) if prev10 else 0
        L = "DOWN" if (current["volume"] - v_avg) < 0 else "UP"
        tr_values = []
        for i in range(1, n - 1):
            high = candles[i]["high"]; low = candles[i]["low"]; prev_close = candles[i - 1]["close"]
            tr_values.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
        atr14 = sum(tr_values[-14:]) / 14 if len(tr_values) >= 14 else None
        P = "invalid"
        if atr14 is not None:
            current_range = current["high"] - current["low"]
            P = "valid" if current_range > atr14 * 1.5 else "invalid"
        A = "invalid"
        if current["volume"] > 0:
            A = "valid" if (abs(current["delta"]) / current["volume"]) > 0.7 else "invalid"
        points_p = 20 if P == "valid" else 0
        points_a = 20 if A == "valid" else 0
        sell_score = (20 if D == "DOWN" else 0) + (20 if L == "DOWN" else 0) + (20 if CDD == "DOWN" else 0) + points_p + points_a
        buy_score = (20 if D == "UP" else 0) + (20 if L == "UP" else 0) + (20 if CDD == "UP" else 0) + points_p + points_a
        if sell_score >= 80 and sell_score > buy_score + 20: direction = "SELL"
        elif buy_score >= 80 and buy_score > sell_score + 20: direction = "BUY"
        elif sell_score >= 60 and sell_score > buy_score: direction = "WEAK_SELL"
        elif buy_score >= 60 and buy_score > sell_score: direction = "WEAK_BUY"
        else: direction = "NEUTRAL"
        return {"direction": direction, "sell_score": sell_score, "buy_score": buy_score,
                "D": D, "L": L, "CDD": CDD, "P": P, "A": A, "points_p": points_p, "points_a": points_a}
    except Exception as e:
        return {"sell_score": 0, "buy_score": 0, "direction": "ERROR", "error": str(e)}


@app.get("/api/direction_score")
async def get_direction_score(interval: str = "1m"):
    result = await asyncio.to_thread(compute_direction_score, current_symbol, interval)
    return {"symbol": current_symbol.upper(), "interval": interval, **result}


@app.get("/api/factor_testing_14")
async def get_factor_testing_14():
    try:
        ticker_data = await http_get_json(
            f"https://api.binance.com/api/v3/ticker/price?symbol={current_symbol.upper()}", timeout=5
        )
        live_price = float(ticker_data["price"]) if ticker_data else None
        async with factor_testing_14_lock:
            factors_copy = {k: dict(v) for k, v in factor_testing_14.items()}
        return {"factors": factors_copy, "timestamp": int(time.time() * 1000),
                "symbol": current_symbol.upper(), "live_price": live_price, "factor_meta": FACTOR_META}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/reset_factor_testing_14")
async def reset_factor_testing_14():
    global factor_testing_14
    async with factor_testing_14_lock:
        for factor in FACTOR_LIST:
            factor_testing_14[factor] = {
                "active": False, "current_direction": "NEUTRAL", "current_signal": "NEUTRAL",
                "entry_price": 0, "entry_time": 0, "win": 0, "loss": 0, "accuracy": 0, "total_trades": 0,
            }
    return {"message": "8-Factor testing data reset"}


@app.get("/api/trade_screenshots")
async def get_trade_screenshots():
    async with lock:
        strategy_shots = [dict(s) for s in trade_screenshots]
        for s in strategy_shots: s["type"] = "strategy"
    async with factor_screenshots_lock:
        factor_shots = [dict(s) for s in factor_screenshots]
        for s in factor_shots: s["type"] = "factor"
    combined = strategy_shots + factor_shots
    combined.sort(key=lambda x: x.get("entry_time", 0), reverse=True)
    return {"available": SCREENSHOTS_AVAILABLE, "screenshots": combined}


class BalanceUpdate(BaseModel):
    balance: float

@app.post("/api/set_balance")
async def set_balance(update: BalanceUpdate):
    global balance
    async with lock:
        balance = update.balance
        save_trading_data()
    return {"balance": balance}


class StrategyPauseUpdate(BaseModel):
    paused: bool

@app.post("/api/strategy_pause")
async def set_strategy_pause(update: StrategyPauseUpdate):
    global strategy_paused
    strategy_paused = bool(update.paused)
    print(f"[Strategy] Pause state set to: {strategy_paused}")
    return {"paused": strategy_paused}


@app.post("/api/strategy_reset")
async def reset_strategy_data():
    global balance, positions, trades_history, trade_screenshots, strategy_paused
    async with lock:
        for shot in trade_screenshots:
            path = shot.get("image_url", "").lstrip("/")
            if path and os.path.exists(path):
                try: os.remove(path)
                except Exception: pass
        balance = DEFAULT_BALANCE
        positions = {}
        trades_history = []
        trade_screenshots = []
        strategy_paused = False
        save_trading_data()
    combined_strategy.trades = []
    combined_strategy.is_active = False
    combined_strategy.position = None
    combined_strategy.current_signal = "NEUTRAL"
    print("Strategy data reset — starting fresh.")
    return {"balance": balance, "message": "Strategy data reset. Starting fresh."}


@app.post("/api/manual_close")
async def manual_close_trade():
    sym = current_symbol.upper()
    if sym not in positions:
        return {"error": "No open position"}
    live_price = await fetch_live_price(sym)
    if live_price is None:
        return {"error": "Could not fetch live price"}
    if combined_strategy.position is not None:
        pnl = 0
        if combined_strategy.position == "LONG":
            pnl = (live_price - combined_strategy.entry_price) / combined_strategy.entry_price * 100
        elif combined_strategy.position == "SHORT":
            pnl = (combined_strategy.entry_price - live_price) / combined_strategy.entry_price * 100
        combined_strategy.trades.append({
            "entry": combined_strategy.entry_price, "exit": live_price, "pnl": pnl,
            "reason": "MANUAL_CLOSE", "entry_time": combined_strategy.entry_time,
            "exit_time": int(time.time() * 1000), "direction": combined_strategy.position,
        })
        combined_strategy.position = None
        combined_strategy.is_active = False
        combined_strategy.current_signal = "NEUTRAL"
    close_position(sym, live_price, "MANUAL_CLOSE")
    print(f"[Combined Strategy] MANUAL CLOSE at {live_price}")
    return {"status": "closed", "price": live_price}


class CombinedSignalStrategy:
    def __init__(self):
        self.current_signal = "NEUTRAL"
        self.position = None
        self.entry_price = 0
        self.entry_time = 0
        self.trades = []
        self.is_active = False

    def update(self, combined_signal, price, timestamp, paused=False):
        if combined_signal == "NEUTRAL":
            self.current_signal = "NEUTRAL"
            return
        if combined_signal == "BUY":
            if self.position == "SHORT":
                self._close_position(price, timestamp, "BUY_SIGNAL")
                if not paused: self._open_position("LONG", price, timestamp)
                else: print("[Combined Strategy] PAUSED – SHORT closed, LONG open skipped")
            elif self.position is None:
                if not paused: self._open_position("LONG", price, timestamp)
                else: print("[Combined Strategy] PAUSED – LONG open skipped")
            self.current_signal = "BUY"
        elif combined_signal == "SELL":
            if self.position == "LONG":
                self._close_position(price, timestamp, "SELL_SIGNAL")
                if not paused: self._open_position("SHORT", price, timestamp)
                else: print("[Combined Strategy] PAUSED – LONG closed, SHORT open skipped")
            elif self.position is None:
                if not paused: self._open_position("SHORT", price, timestamp)
                else: print("[Combined Strategy] PAUSED – SHORT open skipped")
            self.current_signal = "SELL"

    def _open_position(self, direction, price, timestamp):
        global positions
        if self.position is not None:
            self._close_position(price, timestamp, "FORCE_CLOSE")
        self.position = direction
        self.entry_price = price
        self.entry_time = timestamp
        self.is_active = True
        open_position(direction, price, current_symbol.upper(), stop_loss=None, target=None)
        print(f"[Combined Strategy] OPEN {direction} at {price}")

    def _close_position(self, price, timestamp, reason):
        global positions, trades_history
        if self.position is None: return
        close_position(current_symbol.upper(), price, reason)
        pnl = 0
        if self.position == "LONG": pnl = (price - self.entry_price) / self.entry_price * 100
        elif self.position == "SHORT": pnl = (self.entry_price - price) / self.entry_price * 100
        self.trades.append({
            "entry": self.entry_price, "exit": price, "pnl": pnl, "reason": reason,
            "entry_time": self.entry_time, "exit_time": timestamp, "direction": self.position,
        })
        self.position = None
        self.is_active = False
        print(f"[Combined Strategy] CLOSE at {price} | {reason} | P&L: {pnl:.2f}%")

    def get_status(self):
        if self.is_active:
            return {"state": "ACTIVE", "direction": self.position, "entry": self.entry_price, "entry_time": self.entry_time}
        return {"state": "IDLE", "last_signal": self.current_signal}

    def get_stats(self):
        total = len(self.trades)
        if total == 0: return {"total_trades": 0, "win_rate": 0, "avg_pnl": 0}
        wins = sum(1 for t in self.trades if t["pnl"] > 0)
        return {
            "total_trades": total, "wins": wins, "losses": total - wins,
            "win_rate": round((wins / total) * 100, 2),
            "avg_pnl": round(sum(t["pnl"] for t in self.trades) / total, 2),
        }


combined_strategy = CombinedSignalStrategy()


async def fetch_klines(symbol, interval="1m", limit=50):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
    return await http_get_json(url, timeout=10)


async def fetch_live_price(symbol):
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.upper()}"
    data = await http_get_json(url, timeout=8)
    try:
        return float(data["price"]) if data else None
    except Exception:
        return None


def open_position(direction, price, symbol, stop_loss=None, target=None):
    global positions
    size = 1.0
    open_fee = size * FEE_PCT_PER_SIDE
    positions[symbol] = {
        "symbol": symbol, "direction": direction, "entry_price": price,
        "entry_time": int(time.time() * 1000), "size": size,
        "open_fee": round(open_fee, 6), "stop_loss": stop_loss, "target": target,
    }
    save_trading_data()


def close_position(symbol, price, reason="SIGNAL"):
    global positions, balance, trades_history
    pos = positions.get(symbol)
    if pos is None: return
    size = pos["size"]
    entry_price = pos["entry_price"]
    if pos["direction"] == "LONG":
        gross_pnl = size * (price - entry_price) / entry_price
    else:
        gross_pnl = size * (entry_price - price) / entry_price
    close_fee = (size + gross_pnl) * FEE_PCT_PER_SIDE
    net_pnl = gross_pnl - close_fee
    total_fees = pos.get("open_fee", 0) + close_fee
    balance += net_pnl
    trades_history.append({
        "symbol": pos.get("symbol", symbol), "direction": pos["direction"],
        "entry_time": pos["entry_time"], "entry_price": entry_price,
        "exit_time": int(time.time() * 1000), "exit_price": price,
        "size": round(size, 4), "gross_pnl": round(gross_pnl, 4),
        "fees": round(total_fees, 4), "pnl": round(net_pnl, 4),
        "result": "PROFIT" if net_pnl > 0 else ("LOSS" if net_pnl < 0 else "BREAKEVEN"),
        "close_reason": reason, "stop_loss": pos.get("stop_loss"), "target": pos.get("target"),
    })
    del positions[symbol]
    save_trading_data()


async def combined_strategy_loop():
    while True:
        try:
            rows = await fetch_klines(current_symbol, "1m", 50)
            if not rows:
                await asyncio.sleep(1)
                continue
            current_price = float(rows[-1][4])
            async with lock:
                metrics = compute_metrics()
            strength = metrics["box3"] if metrics else "Neutral"
            async with factor_testing_14_lock:
                mean_rev_dir = factor_testing_14["MEAN_REVERSION"]["current_direction"]
                rsi_dir = factor_testing_14["RSI"]["current_direction"]
                z_dir = factor_testing_14["Z_SCORE"]["current_direction"]
            paused = strategy_paused
            buy_signal = (mean_rev_dir == "BUY" and rsi_dir == "BUY" and z_dir == "BUY" and strength == "Very Strong Bullish")
            sell_signal = (mean_rev_dir == "SELL" and rsi_dir == "SELL" and z_dir == "SELL" and strength == "Very Strong Bearish")
            if buy_signal: combined = "BUY"; print(f"[CombinedStrategy] >>> STRICT BUY | Paused={paused}")
            elif sell_signal: combined = "SELL"; print(f"[CombinedStrategy] >>> STRICT SELL | Paused={paused}")
            else: combined = "NEUTRAL"
            combined_strategy.update(combined, current_price, int(time.time() * 1000), paused=paused)
        except Exception as e:
            print(f"Combined Strategy loop error: {e}")
        await asyncio.sleep(1)


@app.get("/api/testing")
async def get_testing():
    async with lock:
        pos_snapshot = {sym: dict(p) for sym, p in positions.items()}
        trades_snapshot = list(trades_history)
        balance_snapshot = balance
    open_positions = []
    for sym, pos in pos_snapshot.items():
        live_price = await fetch_live_price(sym)
        unrealized = 0.0
        if live_price is not None:
            if pos["direction"] == "LONG":
                unrealized = pos["size"] * (live_price - pos["entry_price"]) / pos["entry_price"]
            else:
                unrealized = pos["size"] * (pos["entry_price"] - live_price) / pos["entry_price"]
        open_positions.append({
            "symbol": pos["symbol"], "direction": pos["direction"],
            "entry_price": pos["entry_price"], "entry_time": pos["entry_time"],
            "size": round(pos["size"], 4), "stop_loss": pos.get("stop_loss"),
            "target": pos.get("target"), "current_price": live_price,
            "unrealized_pnl": round(unrealized, 4),
        })
    total_trades = len(trades_snapshot)
    win_trades = sum(1 for t in trades_snapshot if t["pnl"] > 0)
    loss_trades = total_trades - win_trades
    accuracy = round((win_trades / total_trades) * 100, 2) if total_trades > 0 else 0.0
    comb_status = combined_strategy.get_status()
    comb_stats = combined_strategy.get_stats()
    return {
        "balance": round(balance_snapshot, 4),
        "open_positions": open_positions,
        "trades": list(reversed(trades_snapshot)),
        "total_trades": total_trades, "win_trades": win_trades, "loss_trades": loss_trades,
        "accuracy": accuracy,
        "strategy_status": "Combined Signal Strategy running",
        "strategy_paused": strategy_paused,
        "macd_strategy_status": comb_status,
        "macd_strategy_stats": comb_stats,
    }


class FVGArea(BaseModel):
    start_time: int
    end_time: int
    interval: str = "1m"

ALLOWED_FVG_INTERVALS = {"1m", "5m", "1h", "4h", "1d"}
MIN_MIDDLE_RATIO = 1.5


async def fetch_klines_range(symbol, interval, start_ms, end_ms, max_requests=50):
    all_klines = []
    cursor = start_ms
    session = await get_http_session()
    for _ in range(max_requests):
        if cursor >= end_ms: break
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&startTime={cursor}&endTime={end_ms}&limit=1000"
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200: break
                batch = await response.json()
        except Exception as e:
            print(f"[FVG] fetch error: {e}")
            break
        if not batch: break
        all_klines.extend(batch)
        last_open_time = batch[-1][0]
        if last_open_time <= cursor: break
        cursor = last_open_time + 1
        if len(batch) < 1000: break
    return all_klines


@app.post("/api/find_best_fvg")
async def find_best_fvg(area: FVGArea):
    interval = area.interval if area.interval in ALLOWED_FVG_INTERVALS else "1m"
    symbol_at_request = current_symbol
    try:
        klines = await fetch_klines_range(symbol_at_request, interval, area.start_time, area.end_time)
        if not klines:
            return {"error": "Failed to fetch candle data from Binance for this range/timeframe."}
        filtered_candles = [{
            "open": float(k[1]), "high": float(k[2]), "low": float(k[3]), "close": float(k[4]),
            "volume": float(k[5]), "time": k[0]
        } for k in klines]
        if len(filtered_candles) < 3:
            return {"error": f"Only {len(filtered_candles)} candles in this range."}
        start_price = filtered_candles[0]["open"]
        end_price = filtered_candles[-1]["close"]
        is_uptrend = end_price >= start_price
        ref_price = abs(filtered_candles[len(filtered_candles) // 2]["close"])
        if ref_price > 1000: decimals = 2
        elif ref_price > 1: decimals = 4
        elif ref_price > 0.01: decimals = 6
        else: decimals = 8
        all_fvg_zones = []
        max_ratio_seen = 0.0
        for i in range(len(filtered_candles) - 2):
            c1, c2, c3 = filtered_candles[i], filtered_candles[i + 1], filtered_candles[i + 2]
            size1 = c1["high"] - c1["low"]
            size2 = c2["high"] - c2["low"]
            size3 = c3["high"] - c3["low"]
            ratio_seen = round(size2 / max(size1, size3, 1e-12), 2)
            if ratio_seen > max_ratio_seen: max_ratio_seen = ratio_seen
            if size2 >= size1 * MIN_MIDDLE_RATIO and size2 >= size3 * MIN_MIDDLE_RATIO:
                is_middle_green = c2["close"] >= c2["open"]
                if is_uptrend and not is_middle_green: continue
                if not is_uptrend and is_middle_green: continue
                if is_uptrend:
                    fvg_top = round(max(c1["high"], c3["low"]), decimals)
                    fvg_bottom = round(min(c1["high"], c3["low"]), decimals)
                    fvg_type = "BUY"
                else:
                    fvg_top = round(max(c1["low"], c3["high"]), decimals)
                    fvg_bottom = round(min(c1["low"], c3["high"]), decimals)
                    fvg_type = "SELL"
                if fvg_top <= fvg_bottom: continue
                gap_size = round(fvg_top - fvg_bottom, decimals)
                price = round((fvg_top + fvg_bottom) / 2, decimals)
                all_fvg_zones.append({
                    "price": price, "top": fvg_top, "bottom": fvg_bottom,
                    "volume": round(c2["volume"], 4), "type": fvg_type,
                    "window_index": i, "gap_size": gap_size,
                    "trend": "UPTREND" if is_uptrend else "DOWNTREND",
                })
        if not all_fvg_zones:
            return {"error": f"No FVG found. Max ratio: {round(max_ratio_seen, 2)}x.",
                    "candles_fetched": len(filtered_candles), "max_ratio_seen": round(max_ratio_seen, 2)}
        all_fvg_zones.sort(key=lambda x: x["gap_size"], reverse=True)
        best_fvg = all_fvg_zones[0]
        return {
            "best": best_fvg, "total_candidates": len(all_fvg_zones),
            "trend": "UPTREND" if is_uptrend else "DOWNTREND",
            "message": f"{best_fvg['trend']} - FVG @ {best_fvg['price']}"
        }
    except Exception as e:
        print(f"FVG Error: {e}")
        return {"error": f"Error: {str(e)}"}


@app.on_event("startup")
async def startup_event():
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    os.makedirs(FACTOR_SCREENSHOTS_DIR, exist_ok=True)
    if not SCREENSHOTS_AVAILABLE:
        print("matplotlib/mplfinance/pandas not installed — screenshots disabled.")
    load_trading_data()
    await get_http_session()
    asyncio.create_task(depth_listener())
    asyncio.create_task(combined_strategy_loop())
    asyncio.create_task(factor_testing_14_loop())
    asyncio.create_task(gex_background_loop())
    asyncio.create_task(_spoofing_ws_loop())


@app.on_event("shutdown")
async def shutdown_event():
    global _http_session, _spoofing_running
    _spoofing_running = False
    if _http_session is not None and not _http_session.closed:
        await _http_session.close()


@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()