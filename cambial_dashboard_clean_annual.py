# cambial_dashboard_anual.py
# Dashboard Streamlit (corporativo) — Posição (Dia/Mês/Ano) + YoY (média anos anteriores) + Forecast (toggle) + Tema Light/Dark
# Como correr:
#   pip install streamlit pandas numpy altair
#   streamlit run cambial_dashboard_anual.py

import io
import calendar
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


# =============================================================================
# LOGO (preencher)
# =============================================================================
# Pode ser URL (para st.image) ou caminho local (recomendado para page_icon).
LOGO_URL = ""  # <-- preencher


# =============================================================================
# Página (tab / title)
# =============================================================================
# page_icon: Streamlit aceita emoji ou caminho local; algumas versões aceitam URL.
try:
    st.set_page_config(
        page_title="Plataforma Cambial — Executive",
        page_icon=LOGO_URL if LOGO_URL else "📊",
        layout="wide",
    )
except Exception:
    st.set_page_config(
        page_title="Plataforma Cambial — Executive",
        page_icon="📊",
        layout="wide",
    )


# =============================================================================
# Helpers (texto / parsing numérico robusto)
# =============================================================================

_PT_ACCENTS_SRC = "áàâãäéêëíìîïóòôõöúùûüçÁÀÂÃÄÉÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇ"
_PT_ACCENTS_DST = "aaaaaeeeiiiiooooouuuucAAAAAEEEIIIIOOOOOUUUUC"


def _norm_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\xa0", " ")
    s = s.translate(str.maketrans(_PT_ACCENTS_SRC, _PT_ACCENTS_DST))
    s = s.strip().lower()
    s = s.replace("nº", "n").replace("n°", "n")
    for ch in [":", ";", "\t", "\n", "\r"]:
        s = s.replace(ch, " ")
    while "  " in s:
        s = s.replace("  ", " ")
    return s.strip()


def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    """Lê CSV com fallback de encoding + autodetecção de separador."""
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            bio = io.BytesIO(raw)
            df = pd.read_csv(bio, sep=None, engine="python", encoding=enc)
            return df
        except Exception as e:
            last_err = e
    raise last_err


def _pick_col(cols: List[str], must_contain: List[str], occurrence: int = 0) -> Optional[str]:
    hits = []
    for c in cols:
        nc = _norm_text(c)
        ok = True
        for t in must_contain:
            if t not in nc:
                ok = False
                break
        if ok:
            hits.append(c)
    return hits[occurrence] if len(hits) > occurrence else None


def _parse_number_str(x: str) -> Optional[float]:
    """Parse robusto para números PT/EN + símbolos + sufixos K/M/B."""
    if x is None:
        return None
    s = str(x).replace("\xa0", " ").strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None

    # remover símbolos comuns
    for token in ["€", "%", " "]:
        s = s.replace(token, "")

    # sufixos
    mult = 1.0
    s_low = s.lower()
    if s_low.endswith("k"):
        mult = 1e3
        s = s[:-1]
    elif s_low.endswith("m"):
        mult = 1e6
        s = s[:-1]
    elif s_low.endswith("b"):
        mult = 1e9
        s = s[:-1]

    # decidir decimal
    if "." in s and "," in s:
        last_dot = s.rfind(".")
        last_com = s.rfind(",")
        if last_com > last_dot:
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        if "," in s:
            parts = s.split(",")
            if len(parts[-1]) in (1, 2):
                s = s.replace(".", "")
                s = s.replace(",", ".")
            else:
                s = s.replace(",", "")
        else:
            if s.count(".") >= 2:
                s = s.replace(".", "")

    try:
        return float(s) * mult
    except Exception:
        return None


def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).map(_parse_number_str), errors="coerce")


def _to_pct_series(s: pd.Series) -> pd.Series:
    x = _to_float_series(s)
    return np.where(x > 1.0, x / 100.0, x).astype(float)


def _safe_div(a, b):
    try:
        a = float(a)
        b = float(b)
    except Exception:
        return np.nan
    if b == 0 or np.isnan(a) or np.isnan(b):
        return np.nan
    return a / b


def _stock_last(s: pd.Series):
    s2 = pd.to_numeric(s, errors="coerce").dropna()
    return float(s2.iloc[-1]) if len(s2) else np.nan


def _flow_sum(s: pd.Series):
    s2 = pd.to_numeric(s, errors="coerce")
    return float(s2.sum()) if s2.notna().any() else np.nan


def _fmt_int(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{int(round(float(x))):,}".replace(",", " ")


def _fmt_int_compact(x, decimals=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    v = float(x)
    av = abs(v)
    if av >= 1e9:
        return f"{v/1e9:.{decimals}f} B"
    if av >= 1e6:
        return f"{v/1e6:.{decimals}f} M"
    if av >= 1e3:
        return f"{v/1e3:.{decimals}f} K"
    return f"{int(round(v))}"


def _fmt_pct(p, decimals=1):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    return f"{100*float(p):.{decimals}f}%"


def _fmt_eur_compact(x, decimals=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    v = float(x)
    abs_v = abs(v)
    if abs_v >= 1e9:
        return f"€ {v/1e9:.{decimals}f} B"
    if abs_v >= 1e6:
        return f"€ {v/1e6:.{decimals}f} M"
    if abs_v >= 1e3:
        return f"€ {v/1e3:.{decimals}f} K"
    return "€ " + f"{v:,.0f}".replace(",", " ")


def _recompute_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "volume_negocios" in out.columns and "num_operacoes" in out.columns:
        out["ticket_medio"] = out.apply(lambda r: _safe_div(r.get("volume_negocios"), r.get("num_operacoes")), axis=1)
    else:
        out["ticket_medio"] = np.nan

    if "margem_liquida" in out.columns and "num_operacoes" in out.columns:
        out["margem_por_op"] = out.apply(lambda r: _safe_div(r.get("margem_liquida"), r.get("num_operacoes")), axis=1)
    else:
        out["margem_por_op"] = np.nan

    if "margem_liquida" in out.columns and "volume_negocios" in out.columns:
        out["margem_pct_volume"] = out.apply(lambda r: _safe_div(r.get("margem_liquida"), r.get("volume_negocios")), axis=1)
    else:
        out["margem_pct_volume"] = np.nan

    # conversões
    if "conv_ops_s1" not in out.columns or out["conv_ops_s1"].isna().all():
        if "ativados_ops_s1" in out.columns and "clientes_acesso" in out.columns:
            out["conv_ops_s1"] = out.apply(lambda r: _safe_div(r.get("ativados_ops_s1"), r.get("clientes_acesso")), axis=1)

    if "conv_ops_s2" not in out.columns or out["conv_ops_s2"].isna().all():
        if "ativados_ops_s2" in out.columns and "clientes_acesso" in out.columns:
            out["conv_ops_s2"] = out.apply(lambda r: _safe_div(r.get("ativados_ops_s2"), r.get("clientes_acesso")), axis=1)

    return out


# =============================================================================
# Load report (cache)
# =============================================================================

@st.cache_data(show_spinner=False)
def load_report(raw: bytes) -> pd.DataFrame:
    df = _read_csv_bytes(raw)
    df.columns = [str(c).replace("\xa0", " ").strip() for c in df.columns]
    cols = list(df.columns)

    # data
    date_col = None
    for c in cols:
        if _norm_text(c) == "data":
            date_col = c
            break
    if date_col is None:
        date_col = cols[0]

    # map colunas (compatível com o teu header)
    col_clientes = _pick_col(cols, ["clientes", "acesso"], 0)
    col_pend = _pick_col(cols, ["pedidos", "pendentes"], 0)
    col_novos = _pick_col(cols, ["novos", "pedidos"], 0)
    col_desist_total = _pick_col(cols, ["desist", "total"], 0)
    col_desist_ativ = _pick_col(cols, ["de", "ativados"], 0)
    col_desist_pend = _pick_col(cols, ["de", "pendentes"], 0)

    # duas colunas repetidas de "ativados" + "%"
    col_ativ1 = _pick_col(cols, ["clientes", "ativados", "operac"], 0)
    col_pct1 = _pick_col(cols, ["cl", "operac", "acesso"], 0)
    col_ativ2 = _pick_col(cols, ["clientes", "ativados", "operac"], 1)
    col_pct2 = _pick_col(cols, ["cl", "operac", "acesso"], 1)

    col_ops = _pick_col(cols, ["operacoes"], 0)
    col_vol = _pick_col(cols, ["volume"], 0)
    col_marg = _pick_col(cols, ["margem"], 0)

    out = pd.DataFrame()
    out["data"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    def add_num(target: str, src: Optional[str]):
        if src is None or src not in df.columns:
            out[target] = np.nan
        else:
            out[target] = _to_float_series(df[src])

    add_num("clientes_acesso", col_clientes)
    add_num("pedidos_pendentes", col_pend)
    add_num("novos_pedidos", col_novos)
    add_num("desist_total", col_desist_total)
    add_num("desist_ativados", col_desist_ativ)
    add_num("desist_pendentes", col_desist_pend)
    add_num("ativados_ops_s1", col_ativ1)
    add_num("ativados_ops_s2", col_ativ2)
    add_num("num_operacoes", col_ops)

    out["conv_ops_s1"] = _to_pct_series(df[col_pct1]) if col_pct1 and col_pct1 in df.columns else np.nan
    out["conv_ops_s2"] = _to_pct_series(df[col_pct2]) if col_pct2 and col_pct2 in df.columns else np.nan

    out["volume_negocios"] = _to_float_series(df[col_vol]) if col_vol and col_vol in df.columns else np.nan
    out["margem_liquida"] = _to_float_series(df[col_marg]) if col_marg and col_marg in df.columns else np.nan

    out = out.dropna(subset=["data"]).sort_values("data").reset_index(drop=True)
    out = _recompute_derived(out)
    return out


# =============================================================================
# Labels PT
# =============================================================================

_PT_MONTHS = {
    1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"
}


# =============================================================================
# Tema (Light/Dark) — CSS + Altair theme
# =============================================================================

def _apply_theme(theme: str):
    if theme == "Dark":
        bg = "#0E1117"
        panel = "#111827"
        card = "#0B1220"
        border = "rgba(255,255,255,0.10)"
        text = "rgba(255,255,255,0.92)"
        subtle = "rgba(255,255,255,0.70)"
        grid = "rgba(255,255,255,0.12)"
        accent = "#60A5FA"
        accent2 = "#93C5FD"
        good = "#22C55E"
        warn = "#F59E0B"
        bad = "#EF4444"
        df_bg = "#0B1220"
        df_text = "rgba(255,255,255,0.92)"
        df_header = "rgba(255,255,255,0.08)"
    else:
        bg = "#F6F7FB"
        panel = "#FFFFFF"
        card = "#FFFFFF"
        border = "rgba(49,51,63,0.14)"
        text = "rgba(17,24,39,0.95)"
        subtle = "rgba(17,24,39,0.65)"
        grid = "rgba(17,24,39,0.10)"
        accent = "#2563EB"
        accent2 = "#60A5FA"
        good = "#16A34A"
        warn = "#D97706"
        bad = "#DC2626"
        df_bg = "#FFFFFF"
        df_text = "rgba(17,24,39,0.95)"
        df_header = "rgba(17,24,39,0.06)"

    css = f"""
    <style>
      :root {{
        --c-bg: {bg};
        --c-panel: {panel};
        --c-card: {card};
        --c-border: {border};
        --c-text: {text};
        --c-subtle: {subtle};
        --c-accent: {accent};
        --c-accent2: {accent2};
        --c-good: {good};
        --c-warn: {warn};
        --c-bad: {bad};
      }}

      .stApp {{ background: var(--c-bg) !important; color: var(--c-text) !important; }}
      html, body, [class*="css"]  {{ color: var(--c-text) !important; }}
      .block-container {{ padding-top: 0.9rem; padding-bottom: 2.0rem; max-width: 1500px; }}
      footer, #MainMenu {{ visibility: hidden; }}

      h1, h2, h3, h4, h5, h6 {{ letter-spacing: -0.2px; color: var(--c-text) !important; }}
      .subtle {{ color: var(--c-subtle) !important; }}

      .panel {{
        background: var(--c-panel);
        border: 1px solid var(--c-border);
        border-radius: 16px;
        padding: 14px 16px;
      }}

      .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 10px;
      }}
      @media (max-width: 1300px) {{ .kpi-grid {{ grid-template-columns: repeat(3, 1fr); }} }}
      @media (max-width: 800px)  {{ .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }} }}

      .kpi {{
        background: var(--c-card);
        border: 1px solid var(--c-border);
        border-radius: 16px;
        padding: 12px 14px;
        min-height: 80px;
      }}
      .kpi-title {{ font-size: 0.78rem; color: var(--c-subtle); margin-bottom: 4px; }}
      .kpi-value {{ font-size: 1.25rem; font-weight: 760; line-height: 1.15; color: var(--c-text); }}
      .kpi-note  {{ font-size: 0.75rem; color: var(--c-subtle); margin-top: 4px; }}
      .delta-up {{ color: var(--c-good); font-weight: 700; }}
      .delta-down {{ color: var(--c-bad); font-weight: 700; }}
      .delta-flat {{ color: var(--c-subtle); font-weight: 700; }}
      .forecast {{ color: var(--c-accent2); }}

      .badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        border: 1px solid var(--c-border);
        font-size: 0.72rem;
        color: var(--c-subtle);
        margin-left: 8px;
      }}

      /* DataFrame container arredondado + força cores (evita preto no dark) */
      div[data-testid="stDataFrame"] {{
        border-radius: 16px;
        border: 1px solid var(--c-border);
        overflow: hidden;
        background: {df_bg} !important;
        color: {df_text} !important;
      }}
      div[data-testid="stDataFrame"] * {{
        color: {df_text} !important;
      }}
      div[data-testid="stDataFrame"] thead tr th {{
        background: {df_header} !important;
      }}

      /* Botões discretos */
      .stDownloadButton button, .stButton button {{ border-radius: 12px; }}

      /* Topbar */
      .topbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 0.25rem;
      }}
      .topbar-title {{
        font-size: 1.45rem;
        font-weight: 820;
        line-height: 1.1;
      }}
      .topbar-sub {{
        font-size: 0.92rem;
        color: var(--c-subtle);
      }}

      /* Pequeno painel de controlos */
      .ctrl {{
        background: var(--c-panel);
        border: 1px solid var(--c-border);
        border-radius: 16px;
        padding: 12px 12px;
      }}

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    def altair_theme():
        return {
            "config": {
                "background": bg,
                "view": {"stroke": "transparent"},
                "axis": {
                    "labelColor": subtle,
                    "titleColor": subtle,
                    "gridColor": grid,
                    "domainColor": grid,
                    "tickColor": grid,
                    "labelFontSize": 12,
                    "titleFontSize": 12,
                },
                "legend": {
                    "labelColor": subtle,
                    "titleColor": subtle,
                    "labelFontSize": 12,
                    "titleFontSize": 12,
                },
                "title": {"color": subtle, "fontSize": 13},
                "range": {"category": [accent, accent2, good, warn, bad]},
            }
        }

    alt.themes.register("corp_theme", altair_theme)
    alt.themes.enable("corp_theme")

    return {
        "bg": bg,
        "panel": panel,
        "card": card,
        "border": border,
        "text": text,
        "subtle": subtle,
        "grid": grid,
        "accent": accent,
        "accent2": accent2,
        "good": good,
        "warn": warn,
        "bad": bad,
    }


# =============================================================================
# Agregações (mensal / períodos)
# =============================================================================

def build_monthly_year(df_daily: pd.DataFrame, year: int) -> pd.DataFrame:
    dfx = df_daily.copy()
    dfx = dfx[dfx["data"].dt.year == year].sort_values("data")
    if dfx.empty:
        return pd.DataFrame()

    dfx["month_end"] = (dfx["data"] + pd.offsets.MonthEnd(0)).dt.normalize()
    dfx["mes_num"] = dfx["month_end"].dt.month
    dfx["mes"] = dfx["mes_num"].map(_PT_MONTHS)

    stock_cols = [
        "clientes_acesso",
        "pedidos_pendentes",
        "ativados_ops_s1",
        "conv_ops_s1",
        "ativados_ops_s2",
        "conv_ops_s2",
    ]

    flow_cols = [
        "novos_pedidos",
        "desist_total",
        "desist_ativados",
        "desist_pendentes",
        "num_operacoes",
        "volume_negocios",
        "margem_liquida",
    ]

    keep = [c for c in stock_cols + flow_cols if c in dfx.columns]

    rows = []
    for me, g in dfx.groupby("month_end"):
        row = {"data": me, "mes_num": int(me.month), "mes": _PT_MONTHS[int(me.month)]}
        for c in stock_cols:
            if c in keep:
                row[c] = _stock_last(g[c])
        for c in flow_cols:
            if c in keep:
                row[c] = _flow_sum(g[c])
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("data").reset_index(drop=True)
    out = _recompute_derived(out)
    return out


def _month_bounds(year: int, month: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    d0 = pd.Timestamp(year=year, month=month, day=1)
    last_day = calendar.monthrange(year, month)[1]
    d1 = pd.Timestamp(year=year, month=month, day=last_day)
    return d0.normalize(), d1.normalize()


def _period_slice(df_daily: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df_daily[(df_daily["data"] >= start) & (df_daily["data"] <= end)].sort_values("data")


def _period_kpis_from_daily(df_period_daily: pd.DataFrame) -> Dict[str, float]:
    if df_period_daily is None or df_period_daily.empty:
        return {
            "vol": np.nan, "mar": np.nan, "ops": np.nan,
            "clientes_end": np.nan,
            "conv2_end": np.nan,
            "margem_pct": np.nan,
            "ticket": np.nan,
        }

    vol = pd.to_numeric(df_period_daily.get("volume_negocios"), errors="coerce").sum()
    mar = pd.to_numeric(df_period_daily.get("margem_liquida"), errors="coerce").sum()
    ops = pd.to_numeric(df_period_daily.get("num_operacoes"), errors="coerce").sum()

    clientes = pd.to_numeric(df_period_daily.get("clientes_acesso"), errors="coerce").dropna()
    conv2 = pd.to_numeric(df_period_daily.get("conv_ops_s2"), errors="coerce").dropna()

    clientes_end = float(clientes.iloc[-1]) if len(clientes) else np.nan
    conv2_end = float(conv2.iloc[-1]) if len(conv2) else np.nan

    margem_pct = _safe_div(mar, vol)
    ticket = _safe_div(vol, ops)

    return {
        "vol": float(vol) if vol is not None else np.nan,
        "mar": float(mar) if mar is not None else np.nan,
        "ops": float(ops) if ops is not None else np.nan,
        "clientes_end": clientes_end,
        "conv2_end": conv2_end,
        "margem_pct": margem_pct,
        "ticket": ticket,
    }


def _forecast_month_end(df_daily: pd.DataFrame, year: int, month: int, asof: pd.Timestamp) -> Dict[str, float]:
    """Forecast simples: escala linear de fluxos por dias decorridos no mês.
    Stocks: último valor observado até asof.
    """
    m_start, m_end = _month_bounds(year, month)
    asof = min(asof.normalize(), m_end)
    df_mtd = _period_slice(df_daily, m_start, asof)
    k_mtd = _period_kpis_from_daily(df_mtd)

    days_in_month = (m_end - m_start).days + 1
    days_elapsed = (asof - m_start).days + 1
    scale = _safe_div(days_in_month, days_elapsed)

    vol_f = k_mtd["vol"] * scale if not np.isnan(k_mtd["vol"]) else np.nan
    mar_f = k_mtd["mar"] * scale if not np.isnan(k_mtd["mar"]) else np.nan
    ops_f = k_mtd["ops"] * scale if not np.isnan(k_mtd["ops"]) else np.nan

    margem_pct_f = _safe_div(mar_f, vol_f)
    ticket_f = _safe_div(vol_f, ops_f)

    return {
        "vol": vol_f,
        "mar": mar_f,
        "ops": ops_f,
        "clientes_end": k_mtd["clientes_end"],
        "conv2_end": k_mtd["conv2_end"],
        "margem_pct": margem_pct_f,
        "ticket": ticket_f,
        "days_in_month": days_in_month,
        "days_elapsed": days_elapsed,
    }


def _ytd_slice(df_daily: pd.DataFrame, year: int, asof: pd.Timestamp) -> pd.DataFrame:
    y_start = pd.Timestamp(year=year, month=1, day=1)
    return _period_slice(df_daily, y_start, asof.normalize())


def _forecast_year_end(df_daily: pd.DataFrame, year: int, asof: pd.Timestamp) -> Dict[str, float]:
    """Forecast anual com base em sazonalidade histórica (shares cumulativos).
    Fallback: linear por dias no ano.
    """
    asof = asof.normalize()
    month = asof.month

    df_ytd = _ytd_slice(df_daily, year, asof)
    k_ytd = _period_kpis_from_daily(df_ytd)

    years = sorted(df_daily["data"].dt.year.dropna().unique().tolist())
    past_full = []
    for y in years:
        if y >= year:
            continue
        df_m = build_monthly_year(df_daily, y)
        if df_m.empty:
            continue
        # considerar "ano completo" se tiver dados em Dez
        has_dec = pd.to_numeric(df_m.loc[df_m["mes_num"] == 12, "volume_negocios"], errors="coerce").notna().any() or \
                  pd.to_numeric(df_m.loc[df_m["mes_num"] == 12, "margem_liquida"], errors="coerce").notna().any() or \
                  pd.to_numeric(df_m.loc[df_m["mes_num"] == 12, "num_operacoes"], errors="coerce").notna().any()
        if not has_dec:
            continue
        past_full.append((y, df_m))

    shares = []
    for y, df_m in past_full[-5:]:
        vol_total = pd.to_numeric(df_m.get("volume_negocios"), errors="coerce").sum()
        vol_cum = pd.to_numeric(df_m.loc[df_m["mes_num"] <= month, "volume_negocios"], errors="coerce").sum()
        if vol_total and not np.isnan(vol_total) and vol_total > 0:
            shares.append(_safe_div(vol_cum, vol_total))
            continue
        mar_total = pd.to_numeric(df_m.get("margem_liquida"), errors="coerce").sum()
        mar_cum = pd.to_numeric(df_m.loc[df_m["mes_num"] <= month, "margem_liquida"], errors="coerce").sum()
        if mar_total and not np.isnan(mar_total) and mar_total > 0:
            shares.append(_safe_div(mar_cum, mar_total))
            continue
        ops_total = pd.to_numeric(df_m.get("num_operacoes"), errors="coerce").sum()
        ops_cum = pd.to_numeric(df_m.loc[df_m["mes_num"] <= month, "num_operacoes"], errors="coerce").sum()
        if ops_total and not np.isnan(ops_total) and ops_total > 0:
            shares.append(_safe_div(ops_cum, ops_total))

    share_avg = np.nan
    if len(shares):
        shares = [s for s in shares if s is not None and not np.isnan(s) and 0.05 <= s <= 0.95]
        if len(shares):
            share_avg = float(np.mean(shares))

    doy = int(asof.dayofyear)
    days_in_year = 366 if calendar.isleap(year) else 365
    share_linear = _safe_div(doy, days_in_year)

    share_used = share_avg if (share_avg is not None and not np.isnan(share_avg) and share_avg > 0) else share_linear

    def scale_flow(v):
        return _safe_div(v, share_used) if (v is not None and not np.isnan(v) and share_used and share_used > 0) else np.nan

    vol_f = scale_flow(k_ytd["vol"])
    mar_f = scale_flow(k_ytd["mar"])
    ops_f = scale_flow(k_ytd["ops"])

    margem_pct_f = _safe_div(mar_f, vol_f)
    ticket_f = _safe_div(vol_f, ops_f)

    return {
        "vol": vol_f,
        "mar": mar_f,
        "ops": ops_f,
        "clientes_end": k_ytd["clientes_end"],
        "conv2_end": k_ytd["conv2_end"],
        "margem_pct": margem_pct_f,
        "ticket": ticket_f,
        "share_used": share_used,
        "share_mode": "sazonal" if (share_used == share_avg and not np.isnan(share_avg)) else "linear",
        "history_years": [y for y, _ in past_full[-5:]],
    }


# =============================================================================
# Comparações (média anos anteriores)
# =============================================================================

def _avg_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if not dicts:
        return {}
    keys = sorted({k for d in dicts for k in d.keys()})
    out = {}
    for k in keys:
        vals = []
        for d in dicts:
            v = d.get(k, np.nan)
            if v is None:
                continue
            try:
                v = float(v)
            except Exception:
                continue
            if not np.isnan(v):
                vals.append(v)
        out[k] = float(np.mean(vals)) if len(vals) else np.nan
    return out


def _pct_change(a, b):
    if b is None or (isinstance(b, float) and np.isnan(b)) or b == 0 or a is None or (isinstance(a, float) and np.isnan(a)):
        return np.nan
    return (float(a) - float(b)) / float(b)


def _pp_change(a, b):
    if a is None or b is None or (isinstance(a, float) and np.isnan(a)) or (isinstance(b, float) and np.isnan(b)):
        return np.nan
    return float(a) - float(b)


def _same_day_year(year: int, m: int, d: int) -> pd.Timestamp:
    last = calendar.monthrange(year, m)[1]
    return pd.Timestamp(year=year, month=m, day=min(d, last)).normalize()


def _baseline_day(df_daily: pd.DataFrame, sel_year: int, asof: pd.Timestamp, years_all: List[int]) -> Tuple[Dict[str, float], List[int]]:
    m, d = int(asof.month), int(asof.day)
    prev_years = [y for y in years_all if y < sel_year]
    dicts = []
    used = []
    for y in prev_years[-5:]:
        dt = _same_day_year(y, m, d)
        df = df_daily[df_daily["data"] == dt]
        k = _period_kpis_from_daily(df)
        if any([not np.isnan(k.get(x, np.nan)) for x in ("vol", "mar", "ops", "clientes_end")]):
            dicts.append(k)
            used.append(y)
    return _avg_dict(dicts), used


def _baseline_mtd(df_daily: pd.DataFrame, sel_year: int, sel_month: int, asof: pd.Timestamp, years_all: List[int]) -> Tuple[Dict[str, float], List[int]]:
    prev_years = [y for y in years_all if y < sel_year]
    dicts = []
    used = []
    for y in prev_years[-5:]:
        m_start, m_end = _month_bounds(y, sel_month)
        asof_y = _same_day_year(y, sel_month, int(asof.day))
        asof_y = min(asof_y, m_end)
        df = _period_slice(df_daily, m_start, asof_y)
        k = _period_kpis_from_daily(df)
        if any([not np.isnan(k.get(x, np.nan)) for x in ("vol", "mar", "ops", "clientes_end")]):
            dicts.append(k)
            used.append(y)
    return _avg_dict(dicts), used


def _baseline_ytd(df_daily: pd.DataFrame, sel_year: int, asof: pd.Timestamp, years_all: List[int]) -> Tuple[Dict[str, float], List[int]]:
    prev_years = [y for y in years_all if y < sel_year]
    dicts = []
    used = []
    for y in prev_years[-5:]:
        asof_y = _same_day_year(y, int(asof.month), int(asof.day))
        df = _ytd_slice(df_daily, y, asof_y)
        k = _period_kpis_from_daily(df)
        if any([not np.isnan(k.get(x, np.nan)) for x in ("vol", "mar", "ops", "clientes_end")]):
            dicts.append(k)
            used.append(y)
    return _avg_dict(dicts), used


def _baseline_full_month(df_daily: pd.DataFrame, sel_year: int, sel_month: int, years_all: List[int]) -> Tuple[Dict[str, float], List[int]]:
    prev_years = [y for y in years_all if y < sel_year]
    dicts = []
    used = []
    for y in prev_years[-5:]:
        m_start, m_end = _month_bounds(y, sel_month)
        df = _period_slice(df_daily, m_start, m_end)
        k = _period_kpis_from_daily(df)
        if any([not np.isnan(k.get(x, np.nan)) for x in ("vol", "mar", "ops")]):
            dicts.append(k)
            used.append(y)
    return _avg_dict(dicts), used


def _baseline_full_year(df_daily: pd.DataFrame, sel_year: int, years_all: List[int]) -> Tuple[Dict[str, float], List[int]]:
    prev_years = [y for y in years_all if y < sel_year]
    dicts = []
    used = []
    for y in prev_years[-5:]:
        y_start = pd.Timestamp(year=y, month=1, day=1)
        y_end = pd.Timestamp(year=y, month=12, day=31)
        df = _period_slice(df_daily, y_start, y_end)
        k = _period_kpis_from_daily(df)
        if any([not np.isnan(k.get(x, np.nan)) for x in ("vol", "mar", "ops")]):
            dicts.append(k)
            used.append(y)
    return _avg_dict(dicts), used


def _delta_html_pct(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    cls = "delta-flat"
    arrow = "→"
    if p > 0.0005:
        cls = "delta-up"
        arrow = "▲"
    elif p < -0.0005:
        cls = "delta-down"
        arrow = "▼"
    sign = "+" if p >= 0 else ""
    return f"<span class='{cls}'>{arrow} {sign}{p*100:.0f}%</span>"


def _delta_html_pp(pp):
    if pp is None or (isinstance(pp, float) and np.isnan(pp)):
        return ""
    cls = "delta-flat"
    arrow = "→"
    if pp > 0.0005:
        cls = "delta-up"
        arrow = "▲"
    elif pp < -0.0005:
        cls = "delta-down"
        arrow = "▼"
    sign = "+" if pp >= 0 else ""
    return f"<span class='{cls}'>{arrow} {sign}{pp*100:.1f} p.p.</span>"


# =============================================================================
# Charts (Altair)
# =============================================================================

def _chart_monthly_bar(df_month: pd.DataFrame, value_col: str, title: str, y_title: str, color: str) -> alt.Chart:
    if df_month is None or df_month.empty or value_col not in df_month.columns:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_bar().encode(x="x", y="y").properties(height=240)

    d = df_month[["mes_num", "mes", value_col]].copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d[d[value_col].notna()]
    if d.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_bar().encode(x="x", y="y").properties(height=240)

    ch = alt.Chart(d).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, color=color).encode(
        x=alt.X("mes:N", sort=alt.SortField(field="mes_num", order="ascending"), title=None),
        y=alt.Y(f"{value_col}:Q", title=y_title, axis=alt.Axis(grid=True)),
        tooltip=[
            alt.Tooltip("mes:N", title="Mês"),
            alt.Tooltip(f"{value_col}:Q", title=title, format=",.0f"),
        ],
    ).properties(height=240)
    return ch


def _chart_monthly_area_line(df_month: pd.DataFrame, value_col: str, title: str, y_title: str, color: str) -> alt.Chart:
    if df_month is None or df_month.empty or value_col not in df_month.columns:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line().encode(x="x", y="y").properties(height=240)

    d = df_month[["mes_num", "mes", value_col]].copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d[d[value_col].notna()]
    if d.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line().encode(x="x", y="y").properties(height=240)

    area = alt.Chart(d).mark_area(opacity=0.18, interpolate="monotone", color=color).encode(
        x=alt.X("mes:N", sort=alt.SortField(field="mes_num", order="ascending"), title=None),
        y=alt.Y(f"{value_col}:Q", title=y_title, axis=alt.Axis(grid=True)),
    )

    line = alt.Chart(d).mark_line(point=alt.OverlayMarkDef(size=70), interpolate="monotone", color=color).encode(
        x=alt.X("mes:N", sort=alt.SortField(field="mes_num", order="ascending"), title=None),
        y=alt.Y(f"{value_col}:Q", title=y_title),
        tooltip=[
            alt.Tooltip("mes:N", title="Mês"),
            alt.Tooltip(f"{value_col}:Q", title=title, format=",.0f"),
        ],
    )

    return (area + line).properties(height=240)


def _chart_adoption(df_month: pd.DataFrame, accent: str, accent2: str) -> alt.Chart:
    if df_month is None or df_month.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line().encode(x="x", y="y").properties(height=240)

    keep_cols = [c for c in ["mes_num", "mes", "clientes_acesso", "ativados_ops_s2", "conv_ops_s2"] if c in df_month.columns]
    d = df_month[keep_cols].copy()
    for c in ["clientes_acesso", "ativados_ops_s2", "conv_ops_s2"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d[(d.get("clientes_acesso").notna()) | (d.get("ativados_ops_s2").notna()) | (d.get("conv_ops_s2").notna())]
    if d.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line().encode(x="x", y="y").properties(height=240)

    base = alt.Chart(d).encode(
        x=alt.X("mes:N", sort=alt.SortField(field="mes_num", order="ascending"), title=None)
    )

    c1 = base.mark_line(point=True, interpolate="monotone", color=accent2).encode(
        y=alt.Y("clientes_acesso:Q", title="Clientes / Ativados", axis=alt.Axis(grid=True)),
        tooltip=[alt.Tooltip("mes:N", title="Mês"), alt.Tooltip("clientes_acesso:Q", title="Clientes c/ acesso", format=",.0f")]
    )

    c2 = base.mark_line(point=True, interpolate="monotone", color=accent).encode(
        y=alt.Y("ativados_ops_s2:Q", title=None),
        tooltip=[alt.Tooltip("mes:N", title="Mês"), alt.Tooltip("ativados_ops_s2:Q", title="Ativados c/ operações", format=",.0f")]
    )

    c3 = base.mark_line(point=True, interpolate="monotone", strokeDash=[6, 4], color="#22C55E").encode(
        y=alt.Y("conv_ops_s2:Q", title="% clientes com operações", axis=alt.Axis(grid=False, orient="right", format="%"), scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("mes:N", title="Mês"), alt.Tooltip("conv_ops_s2:Q", title="% operações/acesso", format=".1%")]
    )

    return alt.layer(c1, c2, c3).resolve_scale(y="independent").properties(height=260)


# =============================================================================
# UI components
# =============================================================================

def _kpi_grid(items: List[Tuple[str, str, str]]):
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-grid'>", unsafe_allow_html=True)

    for title, value, note in items:
        st.markdown(
            f"""
            <div class="kpi">
              <div class="kpi-title">{title}</div>
              <div class="kpi-value">{value}</div>
              <div class="kpi-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# Carregamento de dados
# =============================================================================

with st.expander("Carregar dados", expanded=True):
    cc1, cc2 = st.columns([2, 1])
    with cc1:
        upl = st.file_uploader("CSV do Report", type=["csv"], label_visibility="collapsed")
    with cc2:
        fallback_path = Path(__file__).with_name("Report.csv")
        use_fallback = False
        if upl is None and fallback_path.exists():
            use_fallback = st.checkbox("Usar Report.csv local", value=True)

raw = None
if upl is not None:
    raw = upl.read()
elif use_fallback:
    raw = fallback_path.read_bytes()

if raw is None:
    st.info("Carregue o CSV para ver o dashboard.")
    st.stop()

try:
    df_daily = load_report(raw)
except Exception as e:
    st.error(f"Erro ao ler o CSV: {e}")
    st.stop()

if df_daily.empty:
    st.warning("CSV sem dados válidos.")
    st.stop()

# Defaults baseados no ficheiro
last_date = df_daily["data"].max().normalize()
years_all = sorted(df_daily["data"].dt.year.dropna().unique().tolist())
last_year = int(last_date.year)
last_month = int(last_date.month)


# =============================================================================
# Estado / Tema
# =============================================================================

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"
if "pos_scope" not in st.session_state:
    st.session_state.pos_scope = "Dia"
if "sel_year" not in st.session_state:
    st.session_state.sel_year = last_year
if "sel_month" not in st.session_state:
    st.session_state.sel_month = last_month
if "asof_date" not in st.session_state:
    st.session_state.asof_date = last_date
if "show_forecast" not in st.session_state:
    st.session_state.show_forecast = False
if "show_month_table" not in st.session_state:
    st.session_state.show_month_table = True
if "show_daily_detail" not in st.session_state:
    st.session_state.show_daily_detail = False

_theme_vars = _apply_theme(st.session_state.theme)


# =============================================================================
# Topo: logo + título
# =============================================================================

c_logo, c_title, c_set = st.columns([0.12, 2.55, 0.55])
with c_logo:
    if LOGO_URL:
        try:
            st.image(LOGO_URL, width=72)
        except Exception:
            pass
with c_title:
    st.markdown(
        """
        <div class='topbar'>
          <div>
            <div class='topbar-title'>Plataforma Cambial — Executive Dashboard</div>
            <div class='topbar-sub'>Foco: crescimento de acesso • adoção/uso • margem. Posição vs média dos anos anteriores + forecast (opcional).</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with c_set:
    with st.popover("⚙️", use_container_width=True):
        st.session_state.theme = st.selectbox("Tema", ["Light", "Dark"], index=1 if st.session_state.theme == "Dark" else 0)
        st.session_state.show_month_table = st.toggle("Mostrar resumo mensal", value=st.session_state.show_month_table)
        st.session_state.show_daily_detail = st.toggle("Mostrar detalhe diário (secundário)", value=st.session_state.show_daily_detail)

# reaplica se mudou
_theme_vars = _apply_theme(st.session_state.theme)


# =============================================================================
# Controlos: Ano/Mês/Data + Posição (Dia/Mês/Ano) + Forecast
# =============================================================================

if len(years_all) == 0:
    st.warning("Sem anos válidos no ficheiro.")
    st.stop()

ctrlL, ctrlR = st.columns([2.2, 1.0])

with ctrlL:
    st.markdown("<div class='ctrl'>", unsafe_allow_html=True)

    r1, r2, r3 = st.columns([1.0, 1.0, 1.2])
    with r1:
        st.session_state.pos_scope = st.radio("Posição", ["Dia", "Mês", "Ano"], horizontal=True, index=["Dia","Mês","Ano"].index(st.session_state.pos_scope))
    with r2:
        st.session_state.sel_year = st.selectbox("Ano", years_all, index=years_all.index(st.session_state.sel_year) if st.session_state.sel_year in years_all else len(years_all)-1)
    with r3:
        sel_year = int(st.session_state.sel_year)
        months_in_year = sorted(df_daily[df_daily["data"].dt.year == sel_year]["data"].dt.month.unique().tolist())
        if len(months_in_year) == 0:
            months_in_year = list(range(1, 13))
        default_month = last_month if sel_year == last_year else (months_in_year[-1] if months_in_year else 12)
        if default_month not in months_in_year:
            default_month = months_in_year[-1]
        st.session_state.sel_month = st.selectbox(
            "Mês",
            months_in_year,
            format_func=lambda m: f"{_PT_MONTHS.get(int(m), str(m))}",
            index=months_in_year.index(st.session_state.sel_month) if st.session_state.sel_month in months_in_year else months_in_year.index(default_month),
            help="Usado para a posição Mês (MTD) e para o resumo mensal."
        )

    # limites de data conforme seleção
    sel_month = int(st.session_state.sel_month)

    def _max_date_in_month(year: int, month: int) -> pd.Timestamp:
        m0, m1 = _month_bounds(year, month)
        dfx = df_daily[(df_daily["data"] >= m0) & (df_daily["data"] <= m1)]
        if dfx.empty:
            return min(last_date, m1)
        return dfx["data"].max().normalize()

    def _max_date_in_year(year: int) -> pd.Timestamp:
        dfx = df_daily[df_daily["data"].dt.year == year]
        if dfx.empty:
            return last_date
        return dfx["data"].max().normalize()

    pos_scope = st.session_state.pos_scope

    if pos_scope == "Dia":
        max_d = _max_date_in_year(sel_year)
        min_d = pd.Timestamp(year=sel_year, month=1, day=1)
    elif pos_scope == "Mês":
        max_d = _max_date_in_month(sel_year, sel_month)
        min_d = pd.Timestamp(year=sel_year, month=sel_month, day=1)
    else:
        max_d = _max_date_in_year(sel_year)
        min_d = pd.Timestamp(year=sel_year, month=1, day=1)

    asof0 = st.session_state.asof_date
    if not isinstance(asof0, pd.Timestamp):
        asof0 = pd.Timestamp(asof0)
    asof0 = asof0.normalize()
    if asof0 < min_d:
        asof0 = min_d
    if asof0 > max_d:
        asof0 = max_d

    dcol1, dcol2 = st.columns([1.2, 1.0])
    with dcol1:
        st.session_state.asof_date = st.date_input(
            "Data de referência",
            value=asof0.date(),
            min_value=min_d.date(),
            max_value=max_d.date(),
            help="Dia: compara com o mesmo dia dos anos anteriores. Mês: compara MTD (1..dia) com anos anteriores. Ano: compara YTD com anos anteriores."
        )

    with dcol2:
        st.markdown(f"<div class='subtle'>Última data no ficheiro: <b>{last_date.date()}</b></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with ctrlR:
    st.markdown("<div class='ctrl'>", unsafe_allow_html=True)
    # Forecast só faz sentido em Mês/Ano
    if st.session_state.pos_scope in ("Mês", "Ano"):
        st.session_state.show_forecast = st.toggle("Mostrar forecast", value=st.session_state.show_forecast)
        st.markdown("<div class='subtle'>Quando ativo: substitui por projeção fim do mês/ano (cor diferente).</div>", unsafe_allow_html=True)
    else:
        st.session_state.show_forecast = False
        st.toggle("Mostrar forecast", value=False, disabled=True)
        st.markdown("<div class='subtle'>Forecast indisponível na posição do dia.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

asof_date = pd.Timestamp(st.session_state.asof_date).normalize()

st.divider()


# =============================================================================
# Posição (quadro principal)
# =============================================================================

sel_year = int(st.session_state.sel_year)
sel_month = int(st.session_state.sel_month)
pos_scope = st.session_state.pos_scope

badge = f"{pos_scope}" + (" • Forecast" if st.session_state.show_forecast else "")

if pos_scope == "Dia":
    d = asof_date
    df_curr = df_daily[df_daily["data"] == d]
    k_curr = _period_kpis_from_daily(df_curr)
    base, used_years = _baseline_day(df_daily, sel_year, d, years_all)
    label = f"{d.date()}"

    vol_d = _pct_change(k_curr["vol"], base.get("vol", np.nan))
    mar_d = _pct_change(k_curr["mar"], base.get("mar", np.nan))
    ops_d = _pct_change(k_curr["ops"], base.get("ops", np.nan))
    cli_d = _pct_change(k_curr["clientes_end"], base.get("clientes_end", np.nan))
    pp_d = _pp_change(k_curr["conv2_end"], base.get("conv2_end", np.nan))

    bench_txt = f"vs média ({', '.join(map(str, used_years))})" if used_years else "vs anos anteriores (sem histórico suficiente)"

    st.markdown(f"### Posição — {label} <span class='badge'>{badge}</span>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtle'>Comparação {bench_txt}. Stocks = valor do dia; Fluxos = o próprio dia.</div>", unsafe_allow_html=True)

    _kpi_grid([
        ("Clientes c/ acesso", _fmt_int(k_curr["clientes_end"]), _delta_html_pct(cli_d) + f" <span class='subtle'>{bench_txt}</span>"),
        ("% clientes com operações", _fmt_pct(k_curr["conv2_end"], 1), _delta_html_pp(pp_d) + f" <span class='subtle'>{bench_txt}</span>"),
        ("Operações (dia)", _fmt_int_compact(k_curr["ops"], 1), _delta_html_pct(ops_d) + f" <span class='subtle'>{bench_txt}</span>"),
        ("Volume (dia)", _fmt_eur_compact(k_curr["vol"], 1), _delta_html_pct(vol_d) + f" <span class='subtle'>{bench_txt}</span>"),
        ("Margem (dia)", _fmt_eur_compact(k_curr["mar"], 1), _delta_html_pct(mar_d) + f" <span class='subtle'>{bench_txt}</span>"),
        ("Ticket médio (dia)", _fmt_eur_compact(k_curr["ticket"], 1), ""),
    ])

elif pos_scope == "Mês":
    m_start, m_end = _month_bounds(sel_year, sel_month)
    asof_m = min(asof_date, m_end)

    df_mtd = _period_slice(df_daily, m_start, asof_m)
    k_mtd = _period_kpis_from_daily(df_mtd)

    if st.session_state.show_forecast:
        k_show = _forecast_month_end(df_daily, sel_year, sel_month, asof_m)
        base_full, used_years = _baseline_full_month(df_daily, sel_year, sel_month, years_all)
        bench_txt = f"vs média mês completo ({', '.join(map(str, used_years))})" if used_years else "vs meses anteriores"
        label = f"{_PT_MONTHS.get(sel_month)} {sel_year} (Forecast)"
        v_class = "forecast"
        meta = f"Forecast: escala linear do MTD ({k_show['days_elapsed']}/{k_show['days_in_month']} dias)."
    else:
        k_show = k_mtd
        base, used_years = _baseline_mtd(df_daily, sel_year, sel_month, asof_m, years_all)
        base_full = base
        bench_txt = f"vs média MTD ({', '.join(map(str, used_years))})" if used_years else "vs MTD anos anteriores"
        label = f"{_PT_MONTHS.get(sel_month)} {sel_year} (MTD até {asof_m.date()})"
        v_class = ""
        meta = f"MTD até {asof_m.date()} (comparação 1..dia com anos anteriores)."

    vol_d = _pct_change(k_show["vol"], base_full.get("vol", np.nan))
    mar_d = _pct_change(k_show["mar"], base_full.get("mar", np.nan))
    ops_d = _pct_change(k_show["ops"], base_full.get("ops", np.nan))
    cli_d = _pct_change(k_show["clientes_end"], base_full.get("clientes_end", np.nan))
    pp_d = _pp_change(k_show["conv2_end"], base_full.get("conv2_end", np.nan))

    st.markdown(f"### Posição — {label} <span class='badge'>{badge}</span>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtle'>{meta} Comparação {bench_txt}.</div>", unsafe_allow_html=True)

    _kpi_grid([
        ("Clientes c/ acesso (último)", f"<span class='{v_class}'>{_fmt_int(k_show['clientes_end'])}</span>", _delta_html_pct(cli_d) + f" <span class='subtle'>{bench_txt}</span>"),
        ("% clientes com operações (último)", f"<span class='{v_class}'>{_fmt_pct(k_show['conv2_end'], 1)}</span>", _delta_html_pp(pp_d) + f" <span class='subtle'>{bench_txt}</span>"),
        ("Operações", f"<span class='{v_class}'>{_fmt_int_compact(k_show['ops'], 1)}</span>", _delta_html_pct(ops_d) + f" <span class='subtle'>{bench_txt}</span>"),
        ("Volume", f"<span class='{v_class}'>{_fmt_eur_compact(k_show['vol'], 1)}</span>", _delta_html_pct(vol_d) + f" <span class='subtle'>{bench_txt}</span>"),
        ("Margem", f"<span class='{v_class}'>{_fmt_eur_compact(k_show['mar'], 1)}</span>", _delta_html_pct(mar_d) + f" <span class='subtle'>{bench_txt}</span>"),
        ("Ticket médio", f"<span class='{v_class}'>{_fmt_eur_compact(k_show['ticket'], 1)}</span>", ""),
    ])

else:  # Ano
    asof_y = asof_date
    df_ytd = _ytd_slice(df_daily, sel_year, asof_y)
    k_ytd = _period_kpis_from_daily(df_ytd)

    if st.session_state.show_forecast:
        k_show = _forecast_year_end(df_daily, sel_year, asof_y)
        base_full, used_years = _baseline_full_year(df_daily, sel_year, years_all)
        bench_txt = f"vs média ano completo ({', '.join(map(str, used_years))})" if used_years else "vs anos anteriores"
        label = f"{sel_year} (Forecast)"
        v_class = "forecast"
        meta = f"Forecast anual via {k_show['share_mode']} (share usado: {k_show['share_used']:.1%})."
    else:
        k_show = k_ytd
        base, used_years = _baseline_ytd(df_daily, sel_year, asof_y, years_all)
        base_full = base
        bench_txt = f"vs média YTD ({', '.join(map(str, used_years))})" if used_years else "vs anos anteriores"
        label = f"{sel_year} (YTD até {asof_y.date()})"
        v_class = ""
        meta = f"YTD até {asof_y.date()} (comparação YTD com anos anteriores)."

    vol_d = _pct_change(k_show["vol"], base_full.get("vol", np.nan))
    mar_d = _pct_change(k_show["mar"], base_full.get("mar", np.nan))
    ops_d = _pct_change(k_show["ops"], base_full.get("ops", np.nan))
    cli_d = _pct_change(k_show["clientes_end"], base_full.get("clientes_end", np.nan))
    pp_d = _pp_change(k_show["conv2_end"], base_full.get("conv2_end", np.nan))

    st.markdown(f"### Posição — {label} <span class='badge'>{badge}</span>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtle'>{meta} Comparação {bench_txt}.</div>", unsafe_allow_html=True)

    _kpi_grid([
        ("Clientes c/ acesso (último)", f"<span class='{v_class}'>{_fmt_int(k_show['clientes_end'])}</span>", _delta_html_pct(cli_d) + f" <span class='subtle'>{bench_txt}</span>"),
        ("% clientes com operações (último)", f"<span class='{v_class}'>{_fmt_pct(k_show['conv2_end'], 1)}</span>", _delta_html_pp(pp_d) + f" <span class='subtle'>{bench_txt}</span>"),
        ("Operações", f"<span class='{v_class}'>{_fmt_int_compact(k_show['ops'], 1)}</span>", _delta_html_pct(ops_d) + f" <span class='subtle'>{bench_txt}</span>"),
        ("Volume", f"<span class='{v_class}'>{_fmt_eur_compact(k_show['vol'], 1)}</span>", _delta_html_pct(vol_d) + f" <span class='subtle'>{bench_txt}</span>"),
        ("Margem", f"<span class='{v_class}'>{_fmt_eur_compact(k_show['mar'], 1)}</span>", _delta_html_pct(mar_d) + f" <span class='subtle'>{bench_txt}</span>"),
        ("Ticket médio", f"<span class='{v_class}'>{_fmt_eur_compact(k_show['ticket'], 1)}</span>", ""),
    ])


# =============================================================================
# Resumo mensal (quadro limpo) — sem linhas vazias + números clean + sinais YoY
# =============================================================================

if st.session_state.show_month_table:
    st.divider()
    st.markdown(f"### {sel_year} — Resumo mensal (limpo)")
    st.markdown("<div class='subtle'>Sem meses vazios. Clientes = número (fim do mês). Adoção = % clientes com operações (fim do mês). Volume/Margem compactos (K/M/B). Δ = vs mesmo mês do ano anterior.</div>", unsafe_allow_html=True)

    df_month = build_monthly_year(df_daily, sel_year)

    if df_month is None or df_month.empty:
        st.info("Sem dados para o ano selecionado.")
    else:
        # manter só meses com algum dado e até ao último mês observado (evita linhas vazias)
        base_num = df_month[[c for c in ["clientes_acesso", "conv_ops_s2", "num_operacoes", "volume_negocios", "margem_liquida"] if c in df_month.columns]].copy()
        keep_any = base_num.apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
        dfm = df_month.loc[keep_any].copy()
        if dfm.empty:
            st.info("Sem dados numéricos no ano selecionado.")
        else:
            max_month_with_data = int(dfm["mes_num"].max())
            dfm = dfm[dfm["mes_num"] <= max_month_with_data].copy()

            # deltas YoY (mês completo) — só se existir ano-1
            has_ly = (sel_year - 1) in years_all
            if has_ly:
                df_ly = build_monthly_year(df_daily, sel_year - 1)
                df_ly = df_ly[["mes_num", "clientes_acesso", "conv_ops_s2", "num_operacoes", "volume_negocios", "margem_liquida"]].copy() if not df_ly.empty else pd.DataFrame()
            else:
                df_ly = pd.DataFrame()

            if not df_ly.empty:
                dfm = dfm.merge(df_ly, on="mes_num", how="left", suffixes=("", "_ly"))
            else:
                for c in ["clientes_acesso_ly", "conv_ops_s2_ly", "num_operacoes_ly", "volume_negocios_ly", "margem_liquida_ly"]:
                    dfm[c] = np.nan

            # formatar
            rows = []
            for _, r in dfm.iterrows():
                mnum = int(r.get("mes_num"))
                mes = _PT_MONTHS.get(mnum, str(mnum))

                cli = r.get("clientes_acesso")
                ado = r.get("conv_ops_s2")
                ops = r.get("num_operacoes")
                vol = r.get("volume_negocios")
                mar = r.get("margem_liquida")

                cli_ly = r.get("clientes_acesso_ly")
                ado_ly = r.get("conv_ops_s2_ly")
                ops_ly = r.get("num_operacoes_ly")
                vol_ly = r.get("volume_negocios_ly")
                mar_ly = r.get("margem_liquida_ly")

                d_cli = _pct_change(cli, cli_ly)
                d_ops = _pct_change(ops, ops_ly)
                d_vol = _pct_change(vol, vol_ly)
                d_mar = _pct_change(mar, mar_ly)
                d_ado_pp = _pp_change(ado, ado_ly)

                rows.append({
                    "Mês": mes,
                    "Clientes c/ acesso": _fmt_int(cli),
                    "% clientes c/ operações": _fmt_pct(ado, 1),
                    "Operações": _fmt_int_compact(ops, 1),
                    "Volume": _fmt_eur_compact(vol, 1),
                    "Margem": _fmt_eur_compact(mar, 1),
                    "Δ Acesso": ("" if not has_ly else ("▲ +0%" if np.isnan(d_cli) else ("▲ " if d_cli>=0 else "▼ ") + ("+" if d_cli>=0 else "") + f"{d_cli*100:.0f}%")),
                    "Δ Adoção": ("" if not has_ly else ("" if np.isnan(d_ado_pp) else ("▲ " if d_ado_pp>=0 else "▼ ") + ("+" if d_ado_pp>=0 else "") + f"{d_ado_pp*100:.1f} p.p.")),
                    "Δ Ops": ("" if not has_ly else ("" if np.isnan(d_ops) else ("▲ " if d_ops>=0 else "▼ ") + ("+" if d_ops>=0 else "") + f"{d_ops*100:.0f}%")),
                    "Δ Volume": ("" if not has_ly else ("" if np.isnan(d_vol) else ("▲ " if d_vol>=0 else "▼ ") + ("+" if d_vol>=0 else "") + f"{d_vol*100:.0f}%")),
                    "Δ Margem": ("" if not has_ly else ("" if np.isnan(d_mar) else ("▲ " if d_mar>=0 else "▼ ") + ("+" if d_mar>=0 else "") + f"{d_mar*100:.0f}%")),
                })

            tbl = pd.DataFrame(rows)

            st.dataframe(
                tbl,
                use_container_width=True,
                hide_index=True,
                height=min(420, 55 + 35 * (len(tbl) + 1)),
                column_config={
                    "Mês": st.column_config.TextColumn(width="small"),
                    "Clientes c/ acesso": st.column_config.TextColumn(width="small"),
                    "% clientes c/ operações": st.column_config.TextColumn(width="small"),
                    "Operações": st.column_config.TextColumn(width="small"),
                    "Volume": st.column_config.TextColumn(width="small"),
                    "Margem": st.column_config.TextColumn(width="small"),
                    "Δ Acesso": st.column_config.TextColumn(width="small"),
                    "Δ Adoção": st.column_config.TextColumn(width="small"),
                    "Δ Ops": st.column_config.TextColumn(width="small"),
                    "Δ Volume": st.column_config.TextColumn(width="small"),
                    "Δ Margem": st.column_config.TextColumn(width="small"),
                }
            )


# =============================================================================
# Gráficos (mensal)
# =============================================================================

st.divider()
st.markdown("### Evolução mensal")
st.markdown("<div class='subtle'>Cores corporativas + meses sem dados filtrados.</div>", unsafe_allow_html=True)

df_month_chart = build_monthly_year(df_daily, sel_year)
if df_month_chart is None or df_month_chart.empty:
    st.info("Sem dados para gráficos.")
else:
    # filtrar meses sem dados
    base_num = df_month_chart[[c for c in ["num_operacoes", "volume_negocios", "margem_liquida", "clientes_acesso"] if c in df_month_chart.columns]].copy()
    keep_any = base_num.apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
    df_month_chart = df_month_chart.loc[keep_any].copy()

    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("**Volume de negócios**")
        st.altair_chart(_chart_monthly_bar(df_month_chart, "volume_negocios", "Volume", "€", _theme_vars["accent"]), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with g2:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("**Nº de operações**")
        st.altair_chart(_chart_monthly_bar(df_month_chart, "num_operacoes", "Operações", "Nº", _theme_vars["accent2"]), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with g3:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("**Margem líquida**")
        st.altair_chart(_chart_monthly_area_line(df_month_chart, "margem_liquida", "Margem", "€", _theme_vars["good"]), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    st.markdown("### Acesso vs Adoção (uso)")
    st.markdown("<div class='subtle'>Clientes com acesso (stock) • Clientes com operações (stock) • % clientes com operações.</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.altair_chart(_chart_adoption(df_month_chart, _theme_vars["accent"], _theme_vars["accent2"]), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# Detalhe diário (opcional)
# =============================================================================

if st.session_state.show_daily_detail:
    st.divider()
    st.markdown("### Detalhe diário (secundário)")
    st.markdown("<div class='subtle'>Últimos 30 dias (charts) + tabela (últimos 60). Útil para validar picos e o ritmo do forecast.</div>", unsafe_allow_html=True)

    d30 = (df_daily["data"] >= (last_date - pd.Timedelta(days=30)))
    df_30 = df_daily.loc[d30].sort_values("data").copy()
    last = df_daily.sort_values("data").iloc[-1]

    k_last = _period_kpis_from_daily(df_daily[df_daily["data"] == last_date])

    _kpi_grid([
        ("Data (último registo)", f"{last_date.date()}", ""),
        ("Clientes com acesso", _fmt_int(k_last.get("clientes_end")), ""),
        ("% clientes com operações", _fmt_pct(k_last.get("conv2_end"), 1), ""),
        ("Operações (dia)", _fmt_int_compact(k_last.get("ops"), 1), ""),
        ("Volume (dia)", _fmt_eur_compact(k_last.get("vol"), 1), ""),
        ("Margem (dia)", _fmt_eur_compact(k_last.get("mar"), 1), ""),
    ])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("**Volume — últimos 30 dias**")
        d = df_30[["data", "volume_negocios"]].copy()
        d["volume_negocios"] = pd.to_numeric(d["volume_negocios"], errors="coerce")
        d = d[d["volume_negocios"].notna()]
        ch = alt.Chart(d).mark_line(interpolate="monotone", color=_theme_vars["accent"]).encode(
            x=alt.X("data:T", title=None),
            y=alt.Y("volume_negocios:Q", title="€"),
            tooltip=[alt.Tooltip("data:T", title="Data"), alt.Tooltip("volume_negocios:Q", title="Volume", format=",.0f")],
        ).properties(height=240)
        st.altair_chart(ch, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("**Operações — últimos 30 dias**")
        d = df_30[["data", "num_operacoes"]].copy()
        d["num_operacoes"] = pd.to_numeric(d["num_operacoes"], errors="coerce")
        d = d[d["num_operacoes"].notna()]
        ch = alt.Chart(d).mark_line(interpolate="monotone", color=_theme_vars["accent2"]).encode(
            x=alt.X("data:T", title=None),
            y=alt.Y("num_operacoes:Q", title="Nº"),
            tooltip=[alt.Tooltip("data:T", title="Data"), alt.Tooltip("num_operacoes:Q", title="Operações", format=",.0f")],
        ).properties(height=240)
        st.altair_chart(ch, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Ver dados diários (últimos 60 dias)", expanded=False):
        df_60 = df_daily[df_daily["data"] >= (last_date - pd.Timedelta(days=60))].copy().sort_values("data")

        rename = {
            "data": "Data",
            "clientes_acesso": "Clientes com acesso",
            "pedidos_pendentes": "Pedidos pendentes",
            "novos_pedidos": "Novos pedidos",
            "desist_total": "Desistências (Total)",
            "desist_ativados": "De Ativados",
            "desist_pendentes": "De Pendentes",
            "ativados_ops_s1": "Ativados c/ operações (S1)",
            "conv_ops_s1": "% ops/acesso (S1)",
            "ativados_ops_s2": "Ativados c/ operações (S2)",
            "conv_ops_s2": "% clientes com operações (S2)",
            "num_operacoes": "Nº operações",
            "volume_negocios": "Volume negócios",
            "margem_liquida": "Margem líquida",
        }

        show = df_60.rename(columns=rename)
        show["Data"] = pd.to_datetime(show["Data"], errors="coerce").dt.date
        st.dataframe(show, use_container_width=True, height=420)


st.caption("Dashboard: Posição (Dia/Mês/Ano) vs média anos anteriores + forecast (opcional) + tema Light/Dark + resumo mensal sem linhas vazias.")
