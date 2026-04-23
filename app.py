"""
Equity Research Platform v2.0
Author: Anshul - CFA Level 3 Candidate
Sections: DCF Valuation, Comparable Valuation, Fundamental Analysis, Risk Analysis, Recommendation
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Optional: auto-refresh
try:
    from streamlit_autorefresh import st_autorefresh
    _AUTOREFRESH = True
except ImportError:
    _AUTOREFRESH = False

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Equity Research Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# GLOBAL CSS
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0d1117; color: #e6edf3; }

div[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 18px;
}
div[data-testid="metric-container"] label { color: #8b949e; font-size: 13px; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-size: 22px; font-weight: 700;
}

.stButton > button {
    background: #238636; color: #fff;
    border: none; border-radius: 8px;
    padding: 8px 20px; font-weight: 600;
    transition: background 0.2s;
}
.stButton > button:hover { background: #2ea043; }

.stDataFrame { border-radius: 10px; overflow: hidden; }

hr { border-color: #30363d; }
h2, h3 { color: #e6edf3 !important; }

.research-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.research-card h4 {
    color: #58a6ff;
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 10px;
    border-bottom: 1px solid #30363d;
    padding-bottom: 6px;
}

.rec-badge-BUY  { background:#0d4b2c; border:1px solid #26a69a;
                  color:#26a69a; border-radius:8px; padding:10px 20px;
                  font-size:24px; font-weight:700; display:inline-block; }
.rec-badge-SELL { background:#4b0d12; border:1px solid #ef5350;
                  color:#ef5350; border-radius:8px; padding:10px 20px;
                  font-size:24px; font-weight:700; display:inline-block; }
.rec-badge-HOLD { background:#2c2c0d; border:1px solid #f4d03f;
                  color:#f4d03f; border-radius:8px; padding:10px 20px;
                  font-size:24px; font-weight:700; display:inline-block; }

.risk-high   { color:#ef5350; font-weight:600; }
.risk-medium { color:#f4d03f; font-weight:600; }
.risk-low    { color:#26a69a; font-weight:600; }

.section-header {
    background: linear-gradient(90deg, #1f2d3d 0%, #161b22 100%);
    border-left: 4px solid #58a6ff;
    border-radius: 6px;
    padding: 10px 16px;
    margin: 18px 0 12px 0;
    font-size: 17px;
    font-weight: 600;
    color: #e6edf3;
}
</style>
""", unsafe_allow_html=True)

if _AUTOREFRESH:
    st_autorefresh(interval=60_000, key="auto_refresh")

# =============================================================================
# SESSION STATE
# =============================================================================
for key, default in [
    ("scan_results", None),
    ("scan_running", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# =============================================================================
# CONSTANTS
# =============================================================================
_PLOT_BG  = "#0d1117"
_GRID_CLR = "#21262d"
_UP_CLR   = "#26a69a"
_DN_CLR   = "#ef5350"
RISK_PCT  = 0.02

SECTOR_PE = {
    "Technology":            28,
    "Financial Services":    18,
    "Consumer Defensive":    35,
    "Consumer Cyclical":     30,
    "Healthcare":            32,
    "Energy":                14,
    "Basic Materials":       15,
    "Industrials":           22,
    "Communication Services":20,
    "Utilities":             18,
    "Real Estate":           25,
    "default":               22,
}

RISK_FREE_RATE   = 0.072
EQUITY_RISK_PREM = 0.055


# =============================================================================
# DATA HELPERS
# =============================================================================

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def safe_download(symbol: str, period: str = "1mo", interval: str = "1d",
                  retries: int = 3) -> pd.DataFrame:
    REQUIRED = {"Open", "High", "Low", "Close"}
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                symbol, period=period, interval=interval,
                progress=False, auto_adjust=True, threads=False,
            )
            if df is None or df.empty:
                raise ValueError("Empty response")
            df = _flatten(df)
            if not REQUIRED.issubset(df.columns):
                raise ValueError("Missing OHLC columns")
            for col in REQUIRED:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df.dropna(subset=list(REQUIRED), how="any", inplace=True)
            if df.empty:
                raise ValueError("All NaN after cleaning")
            return df
        except Exception:
            if attempt < retries:
                time.sleep(attempt * 1.5)
    return pd.DataFrame()


@st.cache_data(ttl=120, show_spinner=False)
def fetch_index_price(symbol: str):
    df = safe_download(symbol, period="5d", interval="1d")
    if df is None or len(df) < 2:
        return None, None, None
    curr = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2])
    chg  = round(curr - prev, 2)
    pct  = round((chg / prev) * 100, 2)
    return round(curr, 2), chg, pct


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker: str) -> pd.DataFrame:
    return safe_download(ticker, period="6mo", interval="1d")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_fundamentals(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_financials(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        return {
            "income":   t.financials,
            "balance":  t.balance_sheet,
            "cashflow": t.cashflow,
            "quarterly_income": t.quarterly_financials,
        }
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nifty500() -> list:
    try:
        url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        df  = pd.read_csv(url)
        return df["Symbol"].tolist()
    except Exception:
        return []


# =============================================================================
# INDICATOR ENGINE
# =============================================================================

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Date" not in df.columns and "Datetime" not in df.columns:
        df = df.reset_index()
    close = pd.to_numeric(df["Close"], errors="coerce")

    df["MA20"]     = close.rolling(20).mean()
    df["MA50"]     = close.rolling(50).mean()
    df["AvgPrice"] = close.rolling(50).mean()

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0.0, float("nan"))
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12             = close.ewm(span=12, adjust=False).mean()
    ema26             = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["Signal_Line"]

    df["H-L"] = df["High"] - df["Low"]
    df["H-C"] = abs(df["High"] - df["Close"].shift(1))
    df["L-C"] = abs(df["Low"]  - df["Close"].shift(1))
    df["TR"]  = df[["H-L", "H-C", "L-C"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(14).mean()

    return df


def generate_signal(df: pd.DataFrame) -> str:
    try:
        last      = df.iloc[-1]
        ma20      = float(last["MA20"])
        ma50      = float(last["MA50"])
        rsi       = float(last["RSI"])
        macd      = float(last["MACD"])
        sig       = float(last["Signal_Line"])
        avg_price = float(last["AvgPrice"])
        cur_price = float(last["Close"])

        if any(pd.isna(v) for v in [ma20, ma50, rsi, macd, sig, avg_price]):
            return "HOLD"

        ma_bull   = ma20 > ma50
        macd_bull = macd > sig
        val_bull  = cur_price < avg_price

        if   ma_bull  and macd_bull  and val_bull:           return "STRONG BUY"
        elif not ma_bull and not macd_bull and not val_bull: return "STRONG SELL"
        elif ma_bull  and val_bull:                          return "BUY"
        elif not ma_bull and not val_bull:                   return "SELL"
        elif rsi < 30 and macd_bull:                         return "BUY"
        elif rsi > 70:                                       return "SELL"
    except Exception:
        pass
    return "HOLD"


# =============================================================================
# DCF VALUATION ENGINE
# =============================================================================

def compute_dcf(info: dict, financials: dict, wacc: float = 0.11,
                terminal_growth: float = 0.04, projection_years: int = 5) -> dict:
    result = {
        "intrinsic_value": None,
        "margin_of_safety": None,
        "base_fcf": None,
        "revenue_cagr": None,
        "wacc": wacc,
        "terminal_growth": terminal_growth,
        "pv_fcfs": [],
        "terminal_value": None,
        "equity_value": None,
        "shares": None,
        "error": None,
    }

    try:
        cf = financials.get("cashflow", pd.DataFrame())
        if cf is None or cf.empty:
            result["error"] = "Cash flow data unavailable"
            return result

        def _get_row(df, candidates):
            for c in candidates:
                if c in df.index:
                    return df.loc[c]
            return None

        op_cf_row = _get_row(cf, ["Operating Cash Flow", "Total Cash From Operating Activities"])
        capex_row = _get_row(cf, ["Capital Expenditure", "Capital Expenditures"])

        if op_cf_row is None:
            result["error"] = "Operating cash flow row not found"
            return result

        op_cf_vals = pd.to_numeric(op_cf_row, errors="coerce").dropna()
        if op_cf_vals.empty:
            result["error"] = "No valid operating cash flow data"
            return result

        capex_vals = (
            pd.to_numeric(capex_row, errors="coerce").dropna()
            if capex_row is not None
            else pd.Series(dtype=float)
        )

        base_op  = float(op_cf_vals.iloc[:2].mean())
        base_cap = float(capex_vals.iloc[:2].mean()) if not capex_vals.empty else 0.0
        base_fcf = base_op + base_cap
        result["base_fcf"] = base_fcf

        inc = financials.get("income", pd.DataFrame())
        rev_row = _get_row(inc, ["Total Revenue", "Revenue"])
        if rev_row is not None:
            rev_vals = pd.to_numeric(rev_row, errors="coerce").dropna()
            if len(rev_vals) >= 2:
                n_yrs = len(rev_vals) - 1
                cagr  = (float(rev_vals.iloc[0]) / float(rev_vals.iloc[-1])) ** (1 / n_yrs) - 1
                cagr  = max(0.02, min(cagr, 0.25))
                result["revenue_cagr"] = cagr
                growth_rate = cagr
            else:
                growth_rate = 0.08
        else:
            growth_rate = 0.08

        projected_fcfs = []
        fcf = base_fcf
        for _ in range(1, projection_years + 1):
            fcf = fcf * (1 + growth_rate)
            projected_fcfs.append(fcf)

        pv_fcfs = [cf / ((1 + wacc) ** t)
                   for t, cf in enumerate(projected_fcfs, start=1)]
        result["pv_fcfs"] = pv_fcfs

        last_fcf = projected_fcfs[-1]
        tv       = last_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
        pv_tv    = tv / ((1 + wacc) ** projection_years)
        result["terminal_value"] = pv_tv

        total_pv = sum(pv_fcfs) + pv_tv

        bal = financials.get("balance", pd.DataFrame())
        cash_row = _get_row(bal, ["Cash And Cash Equivalents",
                                   "Cash Cash Equivalents And Short Term Investments"])
        debt_row = _get_row(bal, ["Total Debt", "Long Term Debt"])

        cash = float(pd.to_numeric(cash_row, errors="coerce").dropna().iloc[0]) if cash_row is not None else 0.0
        debt = float(pd.to_numeric(debt_row, errors="coerce").dropna().iloc[0]) if debt_row is not None else 0.0

        equity_value = total_pv + cash - debt
        result["equity_value"] = equity_value

        shares = info.get("sharesOutstanding", None)
        if not shares or shares == 0:
            result["error"] = "Shares outstanding unavailable"
            return result

        result["shares"] = shares
        intrinsic = equity_value / shares
        result["intrinsic_value"] = intrinsic

        cur_price = info.get("currentPrice") or info.get("regularMarketPrice", None)
        if cur_price and intrinsic:
            mos = ((intrinsic - cur_price) / intrinsic) * 100
            result["margin_of_safety"] = mos

    except Exception as e:
        result["error"] = str(e)

    return result


# =============================================================================
# COMPARABLE VALUATION
# =============================================================================

def compute_comparables(info: dict) -> dict:
    sector   = info.get("sector", "default")
    bench_pe = SECTOR_PE.get(sector, SECTOR_PE["default"])

    trailing_pe = info.get("trailingPE",        None)
    forward_pe  = info.get("forwardPE",          None)
    ev_ebitda   = info.get("enterpriseToEbitda", None)
    pb_ratio    = info.get("priceToBook",        None)
    ps_ratio    = info.get("priceToSalesTrailing12Months", None)
    peg_ratio   = info.get("pegRatio",           None)
    ev          = info.get("enterpriseValue",    None)
    mktcap      = info.get("marketCap",          None)

    verdict         = "FAIRLY VALUED"
    valuation_score = 0

    if trailing_pe and not pd.isna(trailing_pe):
        premium = (trailing_pe - bench_pe) / bench_pe
        if   premium < -0.20: valuation_score += 2
        elif premium < 0:     valuation_score += 1
        elif premium > 0.30:  valuation_score -= 2
        elif premium > 0:     valuation_score -= 1

    if peg_ratio and not pd.isna(peg_ratio):
        if   peg_ratio < 1.0: valuation_score += 2
        elif peg_ratio < 1.5: valuation_score += 1
        elif peg_ratio > 2.5: valuation_score -= 2
        elif peg_ratio > 2.0: valuation_score -= 1

    if   valuation_score >= 2:  verdict = "UNDERVALUED"
    elif valuation_score <= -2: verdict = "OVERVALUED"

    return {
        "trailing_pe": trailing_pe,
        "forward_pe":  forward_pe,
        "ev_ebitda":   ev_ebitda,
        "pb_ratio":    pb_ratio,
        "ps_ratio":    ps_ratio,
        "peg_ratio":   peg_ratio,
        "sector":      sector,
        "bench_pe":    bench_pe,
        "verdict":     verdict,
        "score":       valuation_score,
        "ev":          ev,
        "mktcap":      mktcap,
    }


# =============================================================================
# FUNDAMENTAL ANALYSIS
# =============================================================================

def compute_fundamentals_deep(info: dict, financials: dict) -> dict:
    out = {
        "revenue_series":    [],
        "net_income_series": [],
        "margin_series":     [],
        "eps_series":        [],
        "roe":               None,
        "roce":              None,
        "de_ratio":          None,
        "int_coverage":      None,
        "revenue_cagr":      None,
        "latest_margin":     None,
        "error":             None,
    }

    try:
        inc = financials.get("income",   pd.DataFrame())
        bal = financials.get("balance",  pd.DataFrame())

        def _get_row(df, candidates):
            for c in candidates:
                if c in df.index:
                    return pd.to_numeric(df.loc[c], errors="coerce").dropna()
            return pd.Series(dtype=float)

        rev     = _get_row(inc, ["Total Revenue", "Revenue"])
        ni      = _get_row(inc, ["Net Income", "Net Income Common Stockholders"])
        ebit    = _get_row(inc, ["EBIT", "Operating Income"])
        int_exp = _get_row(inc, ["Interest Expense"])

        if not rev.empty:
            years = [str(d.year) if hasattr(d, "year") else str(d)
                     for d in reversed(rev.index)]
            vals  = [v / 1e7 for v in reversed(rev.values)]
            out["revenue_series"] = list(zip(years, vals))
            if len(rev) >= 2:
                n = len(rev) - 1
                out["revenue_cagr"] = ((float(rev.iloc[0]) / float(rev.iloc[-1])) ** (1 / n) - 1) * 100

        if not ni.empty:
            years_ni = [str(d.year) if hasattr(d, "year") else str(d)
                        for d in reversed(ni.index)]
            vals_ni  = [v / 1e7 for v in reversed(ni.values)]
            out["net_income_series"] = list(zip(years_ni, vals_ni))

            if not rev.empty:
                common_idx = rev.index.intersection(ni.index)
                if not common_idx.empty:
                    margins = [(ni[i] / rev[i]) * 100 for i in common_idx if rev[i] != 0]
                    years_m = [str(d.year) if hasattr(d, "year") else str(d)
                               for d in common_idx]
                    out["margin_series"]  = list(zip(years_m, margins))
                    out["latest_margin"]  = margins[0] if margins else None

        eps = info.get("trailingEps", None)
        if eps:
            out["eps_series"] = [("TTM", eps)]

        equity_row = _get_row(bal, ["Stockholders Equity",
                                     "Total Stockholder Equity",
                                     "Common Stock Equity"])
        if not ni.empty and not equity_row.empty:
            ni_latest = float(ni.iloc[0])
            eq_latest = float(equity_row.iloc[0])
            if eq_latest != 0:
                out["roe"] = (ni_latest / eq_latest) * 100

        assets_row = _get_row(bal, ["Total Assets"])
        cl_row     = _get_row(bal, ["Current Liabilities", "Total Current Liabilities"])
        if not ebit.empty and not assets_row.empty:
            ebit_latest = float(ebit.iloc[0])
            assets      = float(assets_row.iloc[0])
            cl          = float(cl_row.iloc[0]) if not cl_row.empty else 0.0
            cap_emp     = assets - cl
            if cap_emp > 0:
                out["roce"] = (ebit_latest / cap_emp) * 100

        debt_row = _get_row(bal, ["Total Debt", "Long Term Debt"])
        if not debt_row.empty and not equity_row.empty:
            debt   = float(debt_row.iloc[0])
            equity = float(equity_row.iloc[0])
            if equity > 0:
                out["de_ratio"] = debt / equity

        if not ebit.empty and not int_exp.empty:
            e = float(ebit.iloc[0])
            i = abs(float(int_exp.iloc[0]))
            if i > 0:
                out["int_coverage"] = e / i

    except Exception as ex:
        out["error"] = str(ex)

    return out


# =============================================================================
# RISK ANALYSIS FRAMEWORK
# =============================================================================

def assess_risks(info: dict, fund: dict, dcf: dict) -> dict:
    risks = {
        "business":  {"level": "MEDIUM", "points": []},
        "financial": {"level": "MEDIUM", "points": []},
        "macro":     {"level": "MEDIUM", "points": []},
        "valuation": {"level": "MEDIUM", "points": []},
        "overall":   "MEDIUM",
    }

    biz    = risks["business"]
    margin = fund.get("latest_margin")
    cagr   = fund.get("revenue_cagr")

    if margin is not None:
        if margin < 5:
            biz["points"].append("Warning: Thin net margins (<5%) - vulnerable to cost pressures")
            biz["level"] = "HIGH"
        elif margin > 20:
            biz["points"].append("Strong net margins (>20%) - pricing power evident")
            biz["level"] = "LOW"
        else:
            biz["points"].append(f"Moderate net margins (~{margin:.1f}%)")

    if cagr is not None:
        if cagr < 5:
            biz["points"].append("Low revenue CAGR (<5%) - growth may be stagnating")
            if biz["level"] != "HIGH":
                biz["level"] = "MEDIUM"
        elif cagr > 15:
            biz["points"].append(f"Strong revenue CAGR (~{cagr:.1f}%) - growth trajectory intact")
        else:
            biz["points"].append(f"Moderate revenue CAGR (~{cagr:.1f}%)")

    sector = info.get("sector", "")
    if sector in ["Energy", "Basic Materials", "Consumer Cyclical"]:
        biz["points"].append("Cyclical sector - earnings sensitive to economic cycles")
    elif sector in ["Consumer Defensive", "Healthcare", "Utilities"]:
        biz["points"].append("Defensive sector - relatively resilient to downturns")

    if not biz["points"]:
        biz["points"].append("Insufficient data to assess business risk in detail")

    fin = risks["financial"]
    de  = fund.get("de_ratio")
    ic  = fund.get("int_coverage")
    roe = fund.get("roe")

    if de is not None:
        if de > 2.0:
            fin["points"].append(f"High leverage (D/E = {de:.1f}x) - financial stress risk in rate cycles")
            fin["level"] = "HIGH"
        elif de < 0.5:
            fin["points"].append(f"Low leverage (D/E = {de:.1f}x) - strong balance sheet")
            fin["level"] = "LOW"
        else:
            fin["points"].append(f"Moderate leverage (D/E = {de:.1f}x)")

    if ic is not None:
        if ic < 2:
            fin["points"].append(f"Weak interest coverage ({ic:.1f}x) - debt service risk")
            fin["level"] = "HIGH"
        elif ic > 5:
            fin["points"].append(f"Comfortable interest coverage ({ic:.1f}x)")
        else:
            fin["points"].append(f"Adequate interest coverage ({ic:.1f}x)")

    if roe is not None:
        if roe < 10:
            fin["points"].append(f"Low ROE ({roe:.1f}%) - capital allocation may be inefficient")
        elif roe > 20:
            fin["points"].append(f"Strong ROE ({roe:.1f}%) - efficient use of equity capital")

    if not fin["points"]:
        fin["points"].append("Insufficient financial data for detailed assessment")

    macro = risks["macro"]
    beta  = info.get("beta", None)

    if beta is not None:
        if beta > 1.5:
            macro["points"].append(f"High beta ({beta:.2f}) - amplified sensitivity to market moves")
            macro["level"] = "HIGH"
        elif beta < 0.7:
            macro["points"].append(f"Low beta ({beta:.2f}) - defensive, lower market correlation")
            macro["level"] = "LOW"
        else:
            macro["points"].append(f"Beta of {beta:.2f} - broadly in line with market")

    macro["points"].append("RBI rate trajectory and USD/INR dynamics are key macro watchpoints")
    macro["points"].append("Global commodity/oil prices - monitor for input cost impact")

    if sector in ["Information Technology", "Technology"]:
        macro["points"].append("Export-oriented tech - INR appreciation = revenue headwind")
    elif sector in ["Energy", "Basic Materials"]:
        macro["points"].append("Commodity-linked - volatile global pricing risk")

    val = risks["valuation"]
    mos = dcf.get("margin_of_safety")

    if mos is not None:
        if mos < -20:
            val["points"].append(f"Stock trading at significant premium to DCF intrinsic value (MoS = {mos:.1f}%)")
            val["level"] = "HIGH"
        elif mos > 20:
            val["points"].append(f"Meaningful margin of safety vs DCF value (MoS = {mos:.1f}%)")
            val["level"] = "LOW"
        else:
            val["points"].append(f"Modest margin of safety ({mos:.1f}%) - limited downside protection")
    else:
        val["points"].append("DCF data insufficient - valuation risk assessed qualitatively")

    val["points"].append("Any de-rating of sector P/E multiples would compress stock price")

    levels = [risks[k]["level"] for k in ["business", "financial", "macro", "valuation"]]
    if levels.count("HIGH") >= 2:
        risks["overall"] = "HIGH"
    elif levels.count("LOW") >= 3:
        risks["overall"] = "LOW"
    else:
        risks["overall"] = "MEDIUM"

    return risks


# =============================================================================
# INVESTMENT RECOMMENDATION ENGINE
# =============================================================================

def generate_recommendation_full(
    tech_signal: str,
    dcf: dict,
    comparables: dict,
    fund: dict,
    risks: dict,
    info: dict,
    cur_price: float,
) -> dict:
    score   = 0
    reasons = []

    # Technical (20%)
    if "STRONG BUY" in tech_signal:
        score += 2
        reasons.append("Technical: STRONG BUY - MA20 > MA50, MACD bullish crossover, price below 50-day avg")
    elif "BUY" in tech_signal:
        score += 1
        reasons.append("Technical: BUY - price momentum constructive, MACD or RSI supportive")
    elif "STRONG SELL" in tech_signal:
        score -= 2
        reasons.append("Technical: STRONG SELL - bearish across MA, MACD, and relative value")
    elif "SELL" in tech_signal:
        score -= 1
        reasons.append("Technical: SELL - downtrend indicators dominant")
    else:
        reasons.append("Technical: NEUTRAL - mixed signals, no clear directional bias")

    # DCF (40%)
    mos = dcf.get("margin_of_safety")
    if mos is not None:
        if mos > 25:
            score += 2
            reasons.append(f"DCF: Significant undervaluation - margin of safety is {mos:.1f}%")
        elif mos > 5:
            score += 1
            reasons.append(f"DCF: Modest undervaluation - margin of safety is {mos:.1f}%")
        elif mos < -25:
            score -= 2
            reasons.append(f"DCF: Significant overvaluation - stock trades {abs(mos):.1f}% above intrinsic value")
        elif mos < -5:
            score -= 1
            reasons.append(f"DCF: Moderate overvaluation - {abs(mos):.1f}% above intrinsic value")
        else:
            reasons.append(f"DCF: Fairly valued - margin of safety near zero ({mos:.1f}%)")
    else:
        reasons.append("DCF: Insufficient data for DCF-based assessment")

    # Relative Valuation (20%)
    rv_verdict = comparables.get("verdict", "FAIRLY VALUED")
    if rv_verdict == "UNDERVALUED":
        score += 1
        reasons.append(f"Relative Valuation: UNDERVALUED vs sector - P/E and PEG below {comparables.get('sector','') or 'sector'} median")
    elif rv_verdict == "OVERVALUED":
        score -= 1
        reasons.append("Relative Valuation: OVERVALUED vs sector - commands premium without clear catalyst")
    else:
        reasons.append("Relative Valuation: Fairly valued relative to sector peers")

    # Fundamental Quality (20%)
    roe  = fund.get("roe")
    de   = fund.get("de_ratio")
    cagr = fund.get("revenue_cagr")

    fund_score = 0
    if roe  and roe  > 15:  fund_score += 1
    if roe  and roe  < 8:   fund_score -= 1
    if de   and de   > 2:   fund_score -= 1
    if de   and de   < 0.5: fund_score += 1
    if cagr and cagr > 12:  fund_score += 1
    if cagr and cagr < 4:   fund_score -= 1

    score += max(-2, min(2, fund_score))

    roe_str  = f"ROE {roe:.1f}%"       if roe   else "ROE N/A"
    de_str   = f"D/E {de:.1f}x"        if de    else "D/E N/A"
    cagr_str = f"Rev CAGR {cagr:.1f}%" if cagr  else "CAGR N/A"
    reasons.append(f"Fundamentals: {roe_str} | {de_str} | {cagr_str}")

    # Risk adjustment
    if risks.get("overall") == "HIGH":
        score -= 1
        reasons.append("Risk: Overall risk profile is HIGH - warrants additional caution")

    if   score >= 3:  recommendation = "STRONG BUY"
    elif score >= 1:  recommendation = "BUY"
    elif score <= -3: recommendation = "STRONG SELL"
    elif score <= -1: recommendation = "SELL"
    else:             recommendation = "HOLD"

    dcf_iv = dcf.get("intrinsic_value")
    if dcf_iv and dcf_iv > 0:
        target_price = 0.60 * dcf_iv + 0.40 * cur_price
    elif cagr:
        target_price = cur_price * (1 + cagr / 100)
    else:
        target_price = cur_price

    updown_pct = ((target_price - cur_price) / cur_price) * 100

    return {
        "recommendation": recommendation,
        "score":          score,
        "target_price":   target_price,
        "updown_pct":     updown_pct,
        "reasons":        reasons,
    }


# =============================================================================
# CHART BUILDERS
# =============================================================================

def _base_layout() -> dict:
    return dict(
        template="plotly_dark",
        plot_bgcolor=_PLOT_BG,
        paper_bgcolor=_PLOT_BG,
        margin=dict(l=8, r=8, t=32, b=8),
        font=dict(color="#8b949e", size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
        xaxis=dict(showgrid=False, zeroline=False, rangeslider_visible=False),
        yaxis=dict(showgrid=True, gridcolor=_GRID_CLR, zeroline=False),
    )


def build_mini_candle(symbol: str, title: str):
    df = safe_download(symbol, period="1mo", interval="1d")
    if df.empty:
        return None
    fig = go.Figure(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=title,
        increasing_line_color=_UP_CLR, decreasing_line_color=_DN_CLR,
        increasing_fillcolor=_UP_CLR,  decreasing_fillcolor=_DN_CLR,
    ))
    layout = _base_layout()
    layout["height"] = 250
    layout["title"]  = dict(text=title, font_size=13, x=0.02)
    fig.update_layout(**layout)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


def build_analysis_chart(df: pd.DataFrame, ticker: str,
                          stop: float = None, tgt: float = None) -> go.Figure:
    x_col = "Datetime" if "Datetime" in df.columns else "Date"
    if x_col not in df.columns:
        df = df.reset_index()
        x_col = df.columns[0]
    df[x_col] = pd.to_datetime(df[x_col], errors="coerce")
    df = df.dropna(subset=[x_col]).sort_values(x_col).drop_duplicates(x_col)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.60, 0.20, 0.20],
        vertical_spacing=0.03,
    )
    fig.add_trace(go.Candlestick(
        x=df[x_col], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        increasing_line_color=_UP_CLR, decreasing_line_color=_DN_CLR,
        increasing_fillcolor=_UP_CLR,  decreasing_fillcolor=_DN_CLR,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df[x_col], y=df["MA20"], name="MA20",
                              line=dict(color="#f4d03f", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df[x_col], y=df["MA50"], name="MA50",
                              line=dict(color="#5dade2", width=1.5)), row=1, col=1)
    if stop:
        fig.add_hline(y=stop, line_dash="dot", line_color=_DN_CLR, line_width=1,
                      annotation_text="Stop Loss", annotation_font_color=_DN_CLR, row=1, col=1)
    if tgt:
        fig.add_hline(y=tgt, line_dash="dot", line_color=_UP_CLR, line_width=1,
                      annotation_text="Target", annotation_font_color=_UP_CLR, row=1, col=1)
    fig.add_trace(go.Scatter(x=df[x_col], y=df["RSI"], name="RSI",
                              line=dict(color="#ce93d8", width=1.5)), row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color=_UP_CLR, line_width=0.8, row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color=_DN_CLR, line_width=0.8, row=2, col=1)
    hist_colors = [_UP_CLR if v >= 0 else _DN_CLR for v in df["MACD_Hist"]]
    fig.add_trace(go.Bar(x=df[x_col], y=df["MACD_Hist"], name="MACD Hist",
                          marker_color=hist_colors, opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=df[x_col], y=df["MACD"], name="MACD",
                              line=dict(color="#f4d03f", width=1.2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df[x_col], y=df["Signal_Line"], name="Signal",
                              line=dict(color="#ef9a9a", width=1.2)), row=3, col=1)
    layout = _base_layout()
    layout.update(dict(
        height=560,
        title=dict(text=f"{ticker} - Price / RSI / MACD", font_size=14, x=0.02),
        showlegend=True,
    ))
    fig.update_layout(**layout)
    for r in [1, 2, 3]:
        fig.update_xaxes(showgrid=False, zeroline=False,
                          rangebreaks=[dict(bounds=["sat", "mon"])], row=r, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=_GRID_CLR, zeroline=False, row=r, col=1)
    fig.update_yaxes(title_text="RSI",  row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    return fig


def build_financials_chart(fund: dict) -> go.Figure:
    rev_series = fund.get("revenue_series", [])
    mar_series = fund.get("margin_series",  [])

    if not rev_series:
        return None

    rev_years, rev_vals = zip(*rev_series) if rev_series else ([], [])

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=list(rev_years), y=list(rev_vals),
                          name="Revenue (Rs Cr)", marker_color="#58a6ff",
                          opacity=0.75), secondary_y=False)

    ni_series = fund.get("net_income_series", [])
    if ni_series:
        ni_years, ni_vals = zip(*ni_series)
        fig.add_trace(go.Bar(x=list(ni_years), y=list(ni_vals),
                              name="Net Profit (Rs Cr)", marker_color=_UP_CLR,
                              opacity=0.75), secondary_y=False)

    if mar_series:
        mar_years, mar_vals = zip(*mar_series)
        fig.add_trace(go.Scatter(x=list(mar_years), y=list(mar_vals),
                                  name="Net Margin %", mode="lines+markers",
                                  line=dict(color="#f4d03f", width=2),
                                  marker=dict(size=6)), secondary_y=True)

    layout = _base_layout()
    layout["height"]  = 340
    layout["title"]   = dict(text="Revenue / Net Profit / Margin Trend", font_size=13, x=0.02)
    layout["barmode"] = "group"
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Rs Crore",     secondary_y=False,
                     showgrid=True, gridcolor=_GRID_CLR)
    fig.update_yaxes(title_text="Net Margin %", secondary_y=True, showgrid=False)
    return fig


def build_dcf_waterfall(dcf: dict) -> go.Figure:
    pv_fcfs = dcf.get("pv_fcfs", [])
    pv_tv   = dcf.get("terminal_value", 0) or 0
    if not pv_fcfs:
        return None

    scale  = 1e9
    labels = [f"FCF Yr {i+1}" for i in range(len(pv_fcfs))] + ["Terminal Value"]
    values = [v / scale for v in pv_fcfs] + [pv_tv / scale]
    colors = [_UP_CLR] * len(pv_fcfs) + ["#58a6ff"]

    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
                            text=[f"Rs{v:.1f}B" for v in values],
                            textposition="outside"))
    layout = _base_layout()
    layout["height"] = 300
    layout["title"]  = dict(text="DCF: PV of Cash Flows (Rs Billion)", font_size=13, x=0.02)
    fig.update_layout(**layout)
    return fig


# =============================================================================
# SCANNER
# =============================================================================

def _scan_one(symbol: str):
    try:
        df = safe_download(symbol, period="6mo", interval="1d")
        if df.empty or len(df) < 55:
            return None
        df  = compute_indicators(df)
        sig = generate_signal(df)
        if sig == "HOLD":
            return None
        rsi   = df["RSI"].iloc[-1]
        price = df["Close"].iloc[-1]
        return {
            "Signal":    sig,
            "Symbol":    symbol,
            "Price (Rs)": round(float(price), 2),
            "RSI":       round(float(rsi), 2) if not pd.isna(rsi) else None,
        }
    except Exception:
        return None


def run_full_scan(stocks: list, workers: int) -> list:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_scan_one, stocks))
    return [r for r in results if r is not None]


# =============================================================================
# APP LAYOUT
# =============================================================================

st.markdown(
    "<h1 style='margin-bottom:4px'>Equity Research Platform</h1>"
    "<p style='color:#8b949e;margin-top:0'>DCF | Comps | Fundamentals | Risk | Recommendation</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    capital      = st.number_input("Capital (Rs)", value=100_000, step=10_000, min_value=5_000)
    scan_workers = st.slider("Scanner workers", 5, 30, 15, 5)

    st.markdown("---")
    st.markdown("### DCF Assumptions")
    wacc_input = st.slider("WACC (%)", min_value=8, max_value=18, value=11,
                            help="Weighted Average Cost of Capital") / 100
    tgr_input  = st.slider("Terminal Growth Rate (%)", min_value=2, max_value=7, value=4,
                            help="Perpetuity growth rate after projection period") / 100
    proj_yrs   = st.slider("Projection Years", min_value=3, max_value=10, value=5)

    st.markdown("---")
    if _AUTOREFRESH:
        st.caption("Auto-refresh: 60s")
    else:
        st.caption("Install streamlit-autorefresh for live updates.")


# ── SECTION 1 — MARKET OVERVIEW ──────────────────────────────────────────────
st.markdown("## Market Overview")

INDICES = [
    ("^NSEI",    "NIFTY 50"),
    ("^NSEBANK", "BANK NIFTY"),
    ("^BSESN",   "SENSEX"),
]

price_cols = st.columns(3)
for i, (sym, label) in enumerate(INDICES):
    price, chg, pct = fetch_index_price(sym)
    if price:
        price_cols[i].metric(label, f"{price:,.2f}", f"{chg:+.2f}  ({pct:+.2f}%)")
    else:
        price_cols[i].metric(label, "N/A", "Data unavailable")

st.markdown("---")

# ── SECTION 2 — MARKET CHARTS ────────────────────────────────────────────────
st.markdown("## Market Charts — 1-month daily")

chart_cols = st.columns(3)
for i, (sym, label) in enumerate(INDICES):
    with chart_cols[i]:
        fig = build_mini_candle(sym, label)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No chart data for {label}.")

st.markdown("---")

# ── SECTION 3 — STOCK ANALYSIS ───────────────────────────────────────────────
st.markdown("## Stock Analysis")

col_a, col_b = st.columns([3, 1])
ticker = col_a.text_input(
    "Ticker",
    "RELIANCE.NS",
    placeholder="e.g. RELIANCE.NS  TCS.NS  HDFCBANK.NS",
    label_visibility="collapsed",
)
col_b.markdown("<br>", unsafe_allow_html=True)
run_analysis = col_b.button("Analyse", use_container_width=True)

if ticker:
    with st.spinner(f"Loading {ticker}..."):
        raw = fetch_stock_data(ticker)

    if raw.empty or len(raw) < 50:
        st.error(
            f"{ticker} - not enough data (need >= 50 daily bars). "
            "Check the symbol (add .NS for NSE stocks) and try again."
        )
    else:
        data         = compute_indicators(raw)
        last         = data.iloc[-1]
        cur_price    = float(last["Close"])
        atr_val      = float(last["ATR"])         if not pd.isna(last["ATR"])         else 0.0
        rsi_val      = float(last["RSI"])         if not pd.isna(last["RSI"])         else 50.0
        macd_val     = float(last["MACD"])        if not pd.isna(last["MACD"])        else 0.0
        sig_line_val = float(last["Signal_Line"]) if not pd.isna(last["Signal_Line"]) else 0.0

        info   = fetch_fundamentals(ticker)
        signal = generate_signal(data)

        sig_clr   = "#26a69a" if "BUY" in signal else "#ef5350" if "SELL" in signal else "#f4d03f"
        sig_emoji = "BUY" if "BUY" in signal else "SELL" if "SELL" in signal else "HOLD"
        st.markdown(
            f"<div style='background:#161b22;border:1px solid #30363d;"
            f"border-radius:10px;padding:14px 20px;display:inline-block;margin-bottom:12px'>"
            f"<span style='font-size:22px;font-weight:700;color:{sig_clr}'>"
            f"Technical Signal: {signal}</span></div>",
            unsafe_allow_html=True,
        )

        # Risk management
        trade_action = stop_loss = target = None
        qty = 0
        if atr_val > 0 and atr_val >= 0.005 * cur_price:
            if "BUY" in signal:
                trade_action = "BUY"
                stop_loss    = cur_price - 2 * atr_val
                target       = cur_price + 4 * atr_val
            elif "SELL" in signal:
                trade_action = "SELL"
                stop_loss    = cur_price + 2 * atr_val
                target       = cur_price - 4 * atr_val
            if trade_action:
                rps = abs(cur_price - stop_loss)
                if rps > 0:
                    qty = int((capital * RISK_PCT) / rps)
                qty = min(qty, int((capital * 0.10) / max(cur_price, 1)))
                if qty * cur_price < 1_000:
                    trade_action = None
                    qty = 0

        if trade_action and qty > 0:
            t1, t2, t3, t4, t5 = st.columns(5)
            t1.metric("Action",    trade_action)
            t2.metric("Quantity",  f"{qty}")
            t3.metric("Entry",     f"Rs{cur_price:,.2f}")
            t4.metric("Stop Loss", f"Rs{stop_loss:,.2f}")
            t5.metric("Target",    f"Rs{target:,.2f}")
            st.info(
                f"Est. cost: Rs{qty * cur_price:,.0f}  |  "
                f"Risk: Rs{capital * RISK_PCT:,.0f}  |  "
                f"Potential reward: Rs{qty * abs(target - cur_price):,.0f}  |  R:R = 1:2"
            )

        # Technical indicators
        st.markdown("#### Technical Indicators")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RSI (14)",    f"{rsi_val:.2f}",
                  "Oversold" if rsi_val < 30 else "Overbought" if rsi_val > 70 else "Neutral")
        m2.metric("MACD",        f"{macd_val:.4f}")
        m3.metric("Signal Line", f"{sig_line_val:.4f}")
        m4.metric("ATR (14)",    f"Rs{atr_val:.2f}")

        # Price chart
        fig = build_analysis_chart(
            data, ticker,
            stop=stop_loss if trade_action else None,
            tgt=target     if trade_action else None,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── SECTION 3B — DCF ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            "<div class='section-header'>DCF Valuation Model</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Base: Historical Free Cash Flow  |  "
            f"WACC: {wacc_input*100:.1f}%  |  Terminal Growth: {tgr_input*100:.1f}%  |  "
            f"Projection: {proj_yrs} years"
        )

        with st.spinner("Computing DCF..."):
            financials = fetch_financials(ticker)
            dcf_result = compute_dcf(info, financials, wacc=wacc_input,
                                     terminal_growth=tgr_input,
                                     projection_years=proj_yrs)

        if dcf_result["error"]:
            st.warning(f"DCF computation partial: {dcf_result['error']}")

        d1, d2, d3, d4 = st.columns(4)
        iv       = dcf_result.get("intrinsic_value")
        ms       = dcf_result.get("margin_of_safety")
        bcf      = dcf_result.get("base_fcf")
        cagr_dcf = dcf_result.get("revenue_cagr")

        d1.metric("Current Price",       f"Rs{cur_price:,.2f}")
        d2.metric("DCF Intrinsic Value",
                  f"Rs{iv:,.2f}" if iv else "N/A",
                  delta=f"MoS: {ms:.1f}%" if ms is not None else None)
        d3.metric("Base FCF",
                  f"Rs{bcf/1e7:,.1f} Cr" if bcf else "N/A")
        d4.metric("Revenue CAGR (used)",
                  f"{cagr_dcf:.1f}%" if cagr_dcf else "N/A")

        if ms is not None:
            if ms > 20:
                st.success(f"Stock is trading at a {ms:.1f}% discount to intrinsic value - meaningful margin of safety")
            elif ms < -20:
                st.error(f"Stock is {abs(ms):.1f}% above intrinsic value - limited margin of safety")
            else:
                st.info(f"Stock is fairly valued - margin of safety: {ms:.1f}%")

        dcf_fig = build_dcf_waterfall(dcf_result)
        if dcf_fig:
            st.plotly_chart(dcf_fig, use_container_width=True)

        # Sensitivity table
        st.markdown("##### DCF Sensitivity Analysis")
        st.caption("Intrinsic Value per share across WACC and Terminal Growth Rate assumptions")
        wacc_range = [wacc_input - 0.02, wacc_input - 0.01, wacc_input,
                      wacc_input + 0.01, wacc_input + 0.02]
        tgr_range  = [tgr_input - 0.01, tgr_input, tgr_input + 0.01]

        rows = []
        for w in wacc_range:
            row = {"WACC": f"{w*100:.1f}%"}
            for g in tgr_range:
                col_label = f"g={g*100:.1f}%"
                if w <= g or w <= 0 or g <= 0:
                    row[col_label] = "N/A"
                    continue
                r = compute_dcf(info, financials, wacc=w, terminal_growth=g,
                                projection_years=proj_yrs)
                row[col_label] = (
                    f"Rs{r['intrinsic_value']:,.0f}" if r.get("intrinsic_value") else "N/A"
                )
            rows.append(row)

        if rows:
            sens_df = pd.DataFrame(rows).set_index("WACC")
            st.dataframe(sens_df, use_container_width=True)

        # ── SECTION 3C — COMPARABLE VALUATION ───────────────────────────────
        st.markdown("---")
        st.markdown(
            "<div class='section-header'>Comparable / Relative Valuation</div>",
            unsafe_allow_html=True,
        )
        st.caption("Valuation multiples vs sector benchmarks  |  India market context")

        comp = compute_comparables(info)

        def _fmt(v, fmt=".2f"):
            if v is None:
                return "N/A"
            try:
                if isinstance(v, float) and np.isnan(v):
                    return "N/A"
                return f"{v:{fmt}}"
            except Exception:
                return "N/A"

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Trailing P/E", _fmt(comp["trailing_pe"]),
                  delta=f"Sector: {comp['bench_pe']}x")
        c2.metric("Forward P/E",  _fmt(comp["forward_pe"]))
        c3.metric("EV/EBITDA",    _fmt(comp["ev_ebitda"]))
        c4.metric("Price/Book",   _fmt(comp["pb_ratio"]))
        c5.metric("PEG Ratio",    _fmt(comp["peg_ratio"]))

        verdict_clr = {
            "UNDERVALUED":  "#26a69a",
            "OVERVALUED":   "#ef5350",
            "FAIRLY VALUED":"#f4d03f",
        }.get(comp["verdict"], "#f4d03f")

        st.markdown(
            f"<div style='background:#161b22;border:1px solid #30363d;"
            f"border-radius:10px;padding:14px 20px;display:inline-block;margin:8px 0'>"
            f"<span style='font-size:20px;font-weight:700;color:{verdict_clr}'>"
            f"Relative Valuation: {comp['verdict']}</span>"
            f"<span style='color:#8b949e;font-size:13px;margin-left:12px'>"
            f"Sector: {comp['sector'] or 'N/A'}  |  Benchmark P/E: {comp['bench_pe']}x"
            f"</span></div>",
            unsafe_allow_html=True,
        )

        if comp["trailing_pe"] and comp["bench_pe"]:
            try:
                pe_val  = float(comp["trailing_pe"])
                bench   = float(comp["bench_pe"])
                max_val = max(pe_val, bench) * 1.5
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=pe_val,
                    delta={"reference": bench, "valueformat": ".1f"},
                    title={"text": "Trailing P/E vs Sector Median"},
                    gauge={
                        "axis": {"range": [0, max_val]},
                        "bar":  {"color": "#58a6ff"},
                        "steps": [
                            {"range": [0, bench * 0.8],             "color": "#0d4b2c"},
                            {"range": [bench * 0.8, bench * 1.2],   "color": "#2c2c0d"},
                            {"range": [bench * 1.2, max_val],        "color": "#4b0d12"},
                        ],
                        "threshold": {"line": {"color": "#f4d03f", "width": 3},
                                      "thickness": 0.75, "value": bench},
                    },
                ))
                gauge_fig.update_layout(
                    height=260,
                    template="plotly_dark",
                    paper_bgcolor=_PLOT_BG,
                    plot_bgcolor=_PLOT_BG,
                    margin=dict(l=20, r=20, t=50, b=20),
                    font=dict(color="#e6edf3"),
                )
                st.plotly_chart(gauge_fig, use_container_width=True)
            except Exception:
                pass

        # ── SECTION 3D — FUNDAMENTAL ANALYSIS ───────────────────────────────
        st.markdown("---")
        st.markdown(
            "<div class='section-header'>Fundamental Analysis Dashboard</div>",
            unsafe_allow_html=True,
        )

        with st.spinner("Loading financial statements..."):
            fund_deep = compute_fundamentals_deep(info, financials)

        if fund_deep.get("error"):
            st.warning(f"Partial data: {fund_deep['error']}")

        f1, f2, f3, f4, f5 = st.columns(5)
        f1.metric("Revenue CAGR",
                  f"{fund_deep['revenue_cagr']:.1f}%" if fund_deep["revenue_cagr"] else "N/A")
        f2.metric("Net Margin (latest)",
                  f"{fund_deep['latest_margin']:.1f}%" if fund_deep["latest_margin"] else "N/A")
        f3.metric("ROE",
                  f"{fund_deep['roe']:.1f}%" if fund_deep["roe"] else "N/A",
                  delta="Good" if fund_deep["roe"] and fund_deep["roe"] > 15 else None)
        f4.metric("ROCE",
                  f"{fund_deep['roce']:.1f}%" if fund_deep["roce"] else "N/A")
        f5.metric("Debt/Equity",
                  f"{fund_deep['de_ratio']:.2f}x" if fund_deep["de_ratio"] else "N/A",
                  delta="Low" if fund_deep["de_ratio"] and fund_deep["de_ratio"] < 0.5 else None)

        fi1, fi2 = st.columns(2)
        fi1.metric("Interest Coverage",
                   f"{fund_deep['int_coverage']:.1f}x" if fund_deep["int_coverage"] else "N/A",
                   delta="Safe" if fund_deep["int_coverage"] and fund_deep["int_coverage"] > 3 else None)
        fi2.metric("Trailing EPS",
                   f"Rs{info['trailingEps']}" if info.get("trailingEps") else "N/A")

        fin_fig = build_financials_chart(fund_deep)
        if fin_fig:
            st.plotly_chart(fin_fig, use_container_width=True)
        else:
            st.info("Insufficient historical financial data for chart.")

        # ── SECTION 3E — RISK ANALYSIS ───────────────────────────────────────
        st.markdown("---")
        st.markdown(
            "<div class='section-header'>Risk Analysis Framework</div>",
            unsafe_allow_html=True,
        )

        risks = assess_risks(info, fund_deep, dcf_result)

        risk_col1, risk_col2 = st.columns(2)

        def _risk_level_html(level: str) -> str:
            clr = {"HIGH": "#ef5350", "MEDIUM": "#f4d03f", "LOW": "#26a69a"}.get(level, "#8b949e")
            return f"<span style='color:{clr};font-weight:700'>{level}</span>"

        with risk_col1:
            st.markdown(
                f"<div class='research-card'>"
                f"<h4>Business Risk - {_risk_level_html(risks['business']['level'])}</h4>"
                + "".join(f"<p style='margin:4px 0;font-size:14px'>{p}</p>"
                           for p in risks["business"]["points"])
                + "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='research-card'>"
                f"<h4>Macro / Market Risk - {_risk_level_html(risks['macro']['level'])}</h4>"
                + "".join(f"<p style='margin:4px 0;font-size:14px'>{p}</p>"
                           for p in risks["macro"]["points"])
                + "</div>",
                unsafe_allow_html=True,
            )

        with risk_col2:
            st.markdown(
                f"<div class='research-card'>"
                f"<h4>Financial Risk - {_risk_level_html(risks['financial']['level'])}</h4>"
                + "".join(f"<p style='margin:4px 0;font-size:14px'>{p}</p>"
                           for p in risks["financial"]["points"])
                + "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='research-card'>"
                f"<h4>Valuation Risk - {_risk_level_html(risks['valuation']['level'])}</h4>"
                + "".join(f"<p style='margin:4px 0;font-size:14px'>{p}</p>"
                           for p in risks["valuation"]["points"])
                + "</div>",
                unsafe_allow_html=True,
            )

        overall_clr = {"HIGH": "#ef5350", "MEDIUM": "#f4d03f", "LOW": "#26a69a"}.get(
            risks["overall"], "#f4d03f"
        )
        st.markdown(
            f"<div style='text-align:center;padding:14px;background:#161b22;"
            f"border-radius:10px;border:1px solid #30363d;margin-top:4px'>"
            f"<span style='font-size:16px;color:#8b949e'>Overall Risk Profile: </span>"
            f"<span style='font-size:20px;font-weight:700;color:{overall_clr}'>"
            f"{risks['overall']}</span></div>",
            unsafe_allow_html=True,
        )

        # ── SECTION 3F — EQUITY RESEARCH SUMMARY REPORT ─────────────────────
        st.markdown("---")
        st.markdown(
            "<div class='section-header'>Equity Research Summary - Investment Recommendation</div>",
            unsafe_allow_html=True,
        )

        reco = generate_recommendation_full(
            tech_signal=signal,
            dcf=dcf_result,
            comparables=comp,
            fund=fund_deep,
            risks=risks,
            info=info,
            cur_price=cur_price,
        )

        company_name = info.get("longName", ticker)
        sector_name  = info.get("sector",   "N/A")
        industry     = info.get("industry", "N/A")
        mktcap       = info.get("marketCap", None)
        mktcap_str   = f"Rs{mktcap/1e9:,.0f}B" if mktcap else "N/A"
        report_date  = datetime.today().strftime("%d %b %Y")

        st.markdown(
            f"""
            <div class='research-card' style='border-left:4px solid #58a6ff'>
            <div style='display:flex;justify-content:space-between;align-items:flex-start'>
              <div>
                <p style='font-size:22px;font-weight:700;margin:0;color:#e6edf3'>{company_name}</p>
                <p style='font-size:13px;color:#8b949e;margin:4px 0'>
                  {ticker} | {sector_name} | {industry} | Mkt Cap: {mktcap_str}
                </p>
              </div>
              <div style='text-align:right'>
                <p style='font-size:12px;color:#8b949e;margin:0'>Report Date</p>
                <p style='font-size:14px;font-weight:600;margin:2px 0'>{report_date}</p>
              </div>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        rec_key   = reco["recommendation"].replace("STRONG ", "")
        badge_cls = f"rec-badge-{rec_key}"

        rb1, rb2, rb3, rb4 = st.columns([2, 1, 1, 1])
        with rb1:
            st.markdown(
                f"<div class='{badge_cls}'>{reco['recommendation']}</div>",
                unsafe_allow_html=True,
            )
        rb2.metric("Current Price",   f"Rs{cur_price:,.2f}")
        rb3.metric("Target Price",    f"Rs{reco['target_price']:,.2f}",
                   delta=f"{reco['updown_pct']:+.1f}%")
        rb4.metric("Composite Score", f"{reco['score']:+d} / +6",
                   delta="Bullish" if reco["score"] > 0 else "Bearish" if reco["score"] < 0 else "Neutral")

        st.markdown("##### Investment Thesis & Rationale")
        thesis_html = "<div class='research-card'>" + "".join(
            f"<p style='margin:6px 0;font-size:14px;line-height:1.6'>{r}</p>"
            for r in reco["reasons"]
        ) + "</div>"
        st.markdown(thesis_html, unsafe_allow_html=True)

        st.markdown("##### Summary Scorecard")
        scorecard_data = {
            "Dimension": [
                "Technical Analysis",
                "DCF Valuation",
                "Relative Valuation",
                "Fundamental Quality",
                "Risk Profile",
            ],
            "Signal": [
                signal,
                f"MoS: {ms:.1f}%" if ms is not None else "Insufficient Data",
                comp["verdict"],
                (f"ROE {fund_deep['roe']:.1f}% | D/E {fund_deep['de_ratio']:.1f}x"
                 if fund_deep["roe"] and fund_deep["de_ratio"] else "Partial Data"),
                risks["overall"],
            ],
            "Weight": ["20%", "40%", "20%", "20%", "Adjustment"],
        }
        sc_df = pd.DataFrame(scorecard_data)
        st.dataframe(sc_df, use_container_width=True, hide_index=True)

        st.markdown(
            "<div class='research-card' style='border-left:4px solid #f4d03f'>"
            "<h4 style='color:#f4d03f'>Analyst Note</h4>"
            "<p style='font-size:13px;color:#8b949e;line-height:1.7'>"
            "This report is generated by an automated equity research system using publicly available "
            "market data via Yahoo Finance. DCF projections are based on historical cash flows and "
            "represent one valuation scenario - not a guarantee of future performance. "
            "All investments carry risk. This is not financial advice. "
            "Always conduct independent due diligence before investing."
            "</p></div>",
            unsafe_allow_html=True,
        )

st.markdown("---")

# ── SECTION 4 — NIFTY 500 SCANNER ────────────────────────────────────────────
st.markdown("## Nifty 500 Scanner")

stocks_raw = fetch_nifty500()
stocks     = [s + ".NS" for s in stocks_raw]

sc1, sc2 = st.columns([4, 1])
sc1.write(f"Universe: **{len(stocks)}** stocks - Nifty 500")

if sc2.button("Run Scanner", use_container_width=True):
    with st.spinner(f"Scanning {len(stocks)} stocks - this takes 2-4 minutes..."):
        results = run_full_scan(stocks, scan_workers)
    st.session_state["scan_results"] = results
    st.success(f"Scan complete - {len(results)} signals found.")

if st.session_state["scan_results"] is not None:
    results   = st.session_state["scan_results"]
    buy_rows  = sorted(
        [r for r in results if "BUY"  in r["Signal"]],
        key=lambda r: 0 if "STRONG" in r["Signal"] else 1,
    )
    sell_rows = sorted(
        [r for r in results if "SELL" in r["Signal"]],
        key=lambda r: 0 if "STRONG" in r["Signal"] else 1,
    )

    st.write(
        f"**{len(results)}** total signals - "
        f"BUY: **{len(buy_rows)}**  |  SELL: **{len(sell_rows)}**"
    )

    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("### BUY Signals")
        if buy_rows:
            st.dataframe(pd.DataFrame(buy_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No BUY signals found.")
    with rc2:
        st.markdown("### SELL Signals")
        if sell_rows:
            st.dataframe(pd.DataFrame(sell_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No SELL signals found.")

    if st.button("Clear Results"):
        st.session_state["scan_results"] = None
        st.rerun()

st.markdown("---")
st.caption(
    "Data via Yahoo Finance  |  Equity Research Platform v2.0  |  Built by Anshul  |  "
    "Not financial advice. Trade at your own risk."
)