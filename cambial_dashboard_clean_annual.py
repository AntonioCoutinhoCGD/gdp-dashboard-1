# cambial_dashboard_clean_annual.py
# Dashboard Streamlit — Plataforma Cambial (ANUAL por default, sem sidebar, sem alertas)
# Objetivo:
#   - Vista principal: Ano inteiro, comparação mês a mês (tabela estilo report + 2/3 gráficos simples)
#   - Vista secundária: "Hoje" (diário) para ver o estado atual da plataforma
#   - ZERO sidebar
#   - ZERO alertas
#   - ZERO nomes internos feios na UI (sem clientes_acesso, num_operacoes, etc.)
# Como correr:
#   pip install streamlit pandas numpy
#   streamlit run cambial_dashboard_clean_annual.py

import io
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# Página + CSS (limpo)
# =============================================================================

st.set_page_config(
    page_title="Plataforma Cambial — Anual",
    page_icon="📊",
    layout="wide",
)

_CSS = """
<style>
  .block-container { padding-top: 1.0rem; padding-bottom: 1.6rem; max-width: 1400px; }
  footer, #MainMenu { visibility: hidden; }

  h1 { letter-spacing: -0.3px; margin-bottom: 0.15rem; }
  h2, h3 { letter-spacing: -0.2px; }
  .subtle { opacity: 0.82; margin-top: -0.25rem; }

  /* Tabela estilo report */
  .report-wrap { overflow-x: auto; border: 1px solid rgba(49, 51, 63, 0.15); border-radius: 12px; }
  table.report {
    border-collapse: collapse;
    width: 100%;
    min-width: 980px;
    font-size: 0.92rem;
  }
  table.report th, table.report td {
    border: 1px solid rgba(49, 51, 63, 0.12);
    padding: 8px 10px;
    text-align: center;
    white-space: nowrap;
  }
  table.report thead th {
    background: rgba(49, 51, 63, 0.06);
    font-weight: 700;
  }
  table.report td.rowhdr {
    text-align: left;
    font-weight: 600;
    background: rgba(49, 51, 63, 0.03);
    position: sticky;
    left: 0;
    z-index: 2;
  }
  table.report th.corner {
    text-align: left;
    position: sticky;
    left: 0;
    z-index: 3;
    background: rgba(49, 51, 63, 0.06);
  }
  tr.sep-row td { background: rgba(49, 51, 63, 0.05); height: 8px; padding: 6px 0; }
  tr.title-row td { background: rgba(49, 51, 63, 0.08); font-weight: 800; text-transform: uppercase; text-align: left; }

  /* Cards do "Hoje" */
  .mini-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
  @media (max-width: 1100px) { .mini-grid { grid-template-columns: repeat(2, 1fr); } }
  .mini-card {
    border: 1px solid rgba(49,51,63,0.15);
    border-radius: 12px;
    padding: 10px 12px;
    background: rgba(255,255,255,0.75);
  }
  .mini-title { font-size: 0.82rem; opacity: 0.85; margin-bottom: 4px; }
  .mini-value { font-size: 1.22rem; font-weight: 800; line-height: 1.1; }
  .mini-note { font-size: 0.78rem; opacity: 0.75; margin-top: 4px; }
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)

# =============================================================================
# Helpers
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
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            bio = io.BytesIO(raw)
            return pd.read_csv(bio, sep=None, engine="python", encoding=enc)
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


def _to_float_series(s: pd.Series) -> pd.Series:
    """Robusto para números com €, espaços, % e separadores pt/en."""
    out = s.astype(str).str.replace("\xa0", " ", regex=False).str.strip()
    out = out.str.replace("€", "", regex=False)
    out = out.str.replace(" ", "", regex=False)
    out = out.str.replace("%", "", regex=False)

    # Caso comum PT: 40,359 -> 40359 ; 75.801 -> 75801
    # Estratégia: remover separador de milhar (.) e trocar decimal vírgula por ponto.
    out = out.str.replace(".", "", regex=False)
    out = out.str.replace(",", ".", regex=False)

    return pd.to_numeric(out, errors="coerce")


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


def _last_non_null(s: pd.Series):
    s2 = pd.to_numeric(s, errors="coerce").dropna()
    return float(s2.iloc[-1]) if len(s2) else np.nan


def _sum_or_nan(s: pd.Series):
    s2 = pd.to_numeric(s, errors="coerce")
    return float(s2.sum()) if s2.notna().any() else np.nan


def _fmt_int(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{int(round(float(x))):,}".replace(",", " ")


def _fmt_pct(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return f"{100*float(p):.0f}%"


def _fmt_eur_compact(x, decimals=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    v = float(x)
    av = abs(v)
    if av >= 1e9:
        return f"€ {v/1e9:.{decimals}f} B"
    if av >= 1e6:
        return f"€ {v/1e6:.{decimals}f} M"
    if av >= 1e3:
        return f"€ {v/1e3:.{decimals}f} K"
    return "€ " + f"{v:,.0f}".replace(",", " ")


def _fmt_eur_full(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    v = float(x)
    return "€ " + f"{v:,.0f}".replace(",", " ")


def _recompute_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "Volume negócios" in out.columns and "Nº operações" in out.columns:
        out["Ticket médio"] = out.apply(lambda r: _safe_div(r.get("Volume negócios"), r.get("Nº operações")), axis=1)
    else:
        out["Ticket médio"] = np.nan

    if "Margem líquida" in out.columns and "Nº operações" in out.columns:
        out["Margem por operação"] = out.apply(lambda r: _safe_div(r.get("Margem líquida"), r.get("Nº operações")), axis=1)
    else:
        out["Margem por operação"] = np.nan

    if "Margem líquida" in out.columns and "Volume negócios" in out.columns:
        out["Margem/Volume"] = out.apply(lambda r: _safe_div(r.get("Margem líquida"), r.get("Volume negócios")), axis=1)
    else:
        out["Margem/Volume"] = np.nan

    # conversões: se faltarem, calcula
    if "% operações / acesso (Série 1)" not in out.columns or out["% operações / acesso (Série 1)"].isna().all():
        if "Ativados c/ operações (Série 1)" in out.columns and "Nº clientes com acesso" in out.columns:
            out["% operações / acesso (Série 1)"] = out.apply(
                lambda r: _safe_div(r.get("Ativados c/ operações (Série 1)"), r.get("Nº clientes com acesso")), axis=1
            )

    if "% operações / acesso (Série 2)" not in out.columns or out["% operações / acesso (Série 2)"].isna().all():
        if "Ativados c/ operações (Série 2)" in out.columns and "Nº clientes com acesso" in out.columns:
            out["% operações / acesso (Série 2)"] = out.apply(
                lambda r: _safe_div(r.get("Ativados c/ operações (Série 2)"), r.get("Nº clientes com acesso")), axis=1
            )

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

    # map (pela semântica do teu header)
    col_clientes = _pick_col(cols, ["clientes", "acesso"], 0)
    col_pend = _pick_col(cols, ["pedidos", "pendentes"], 0)
    col_novos = _pick_col(cols, ["novos", "pedidos"], 0)
    col_desist_total = _pick_col(cols, ["desist", "total"], 0)
    col_desist_ativ = _pick_col(cols, ["de", "ativados"], 0)
    col_desist_pend = _pick_col(cols, ["de", "pendentes"], 0)

    # 2 séries de ativados + %
    col_ativ1 = _pick_col(cols, ["clientes", "ativados", "operac"], 0)
    col_pct1 = _pick_col(cols, ["cl", "operac", "acesso"], 0)
    col_ativ2 = _pick_col(cols, ["clientes", "ativados", "operac"], 1)
    col_pct2 = _pick_col(cols, ["cl", "operac", "acesso"], 1)

    col_ops = _pick_col(cols, ["operacoes"], 0)
    col_vol = _pick_col(cols, ["volume"], 0)
    col_marg = _pick_col(cols, ["margem"], 0)

    out = pd.DataFrame()
    out["Data"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    def add_num(target: str, src: Optional[str]):
        if src is None or src not in df.columns:
            out[target] = np.nan
        else:
            out[target] = _to_float_series(df[src])

    # Nomes LIMPOS já aqui
    add_num("Nº clientes com acesso", col_clientes)
    add_num("Nº pedidos pendentes", col_pend)
    add_num("Novos pedidos", col_novos)
    add_num("Desistências (Total)", col_desist_total)
    add_num("De Ativados", col_desist_ativ)
    add_num("De Pendentes", col_desist_pend)

    add_num("Ativados c/ operações (Série 1)", col_ativ1)
    add_num("Ativados c/ operações (Série 2)", col_ativ2)

    out["% operações / acesso (Série 1)"] = _to_pct_series(df[col_pct1]) if col_pct1 and col_pct1 in df.columns else np.nan
    out["% operações / acesso (Série 2)"] = _to_pct_series(df[col_pct2]) if col_pct2 and col_pct2 in df.columns else np.nan

    add_num("Nº operações", col_ops)
    add_num("Volume negócios", col_vol)
    add_num("Margem líquida", col_marg)

    out = out.dropna(subset=["Data"]).sort_values("Data").reset_index(drop=True)
    out = _recompute_derived(out)

    return out


# =============================================================================
# Agregação mensal (ano)
# =============================================================================

_PT_MONTHS = {
    1: "jan", 2: "fev", 3: "mar", 4: "abr", 5: "mai", 6: "jun",
    7: "jul", 8: "ago", 9: "set", 10: "out", 11: "nov", 12: "dez"
}


def _month_label(dt: pd.Timestamp) -> str:
    # 31-jan / 28-fev / ...
    return f"{dt.day:02d}-{_PT_MONTHS[int(dt.month)]}"


def _looks_cumulative(s: pd.Series) -> bool:
    """Heurística: se a série for maioritariamente não-decrescente, assume acumulado (usar último do mês)."""
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) < 8:
        return False
    diffs = x.diff().dropna()
    if len(diffs) == 0:
        return False
    frac_nonneg = (diffs >= 0).mean()
    return frac_nonneg >= 0.80


def build_monthly_year(df_daily: pd.DataFrame, year: int) -> pd.DataFrame:
    d = df_daily.copy()
    d = d[d["Data"].dt.year == year].sort_values("Data")
    if d.empty:
        return pd.DataFrame()

    d["Mês"] = (d["Data"] + pd.offsets.MonthEnd(0)).dt.normalize()

    # Definição base:
    stock_cols = [
        "Nº clientes com acesso",
        "Nº pedidos pendentes",
        "Ativados c/ operações (Série 1)",
        "% operações / acesso (Série 1)",
        "Ativados c/ operações (Série 2)",
        "% operações / acesso (Série 2)",
    ]

    flow_cols = [
        "Novos pedidos",
        "Desistências (Total)",
        "De Ativados",
        "De Pendentes",
    ]

    # Estes 3 variam conforme o ficheiro (pode ser mensal ou acumulado). Decidimos por heurística.
    adaptive_cols = [
        "Nº operações",
        "Volume negócios",
        "Margem líquida",
    ]

    # decidir para o ano
    adaptive_as_stock = {c: _looks_cumulative(d[c]) for c in adaptive_cols if c in d.columns}

    rows = []
    for me, g in d.groupby("Mês"):
        row = {"Data": me}

        # stocks
        for c in stock_cols:
            if c in g.columns:
                row[c] = _last_non_null(g[c])

        # flows
        for c in flow_cols:
            if c in g.columns:
                row[c] = _sum_or_nan(g[c])

        # adaptativos
        for c in adaptive_cols:
            if c in g.columns:
                if adaptive_as_stock.get(c, False):
                    row[c] = _last_non_null(g[c])
                else:
                    row[c] = _sum_or_nan(g[c])

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("Data").reset_index(drop=True)
    out = _recompute_derived(out)

    # garantir 12 meses
    months = pd.date_range(start=f"{year}-01-31", end=f"{year}-12-31", freq="ME")
    base = pd.DataFrame({"Data": months})
    out = base.merge(out, on="Data", how="left")

    return out


# =============================================================================
# Construir tabela report (rows x months)
# =============================================================================


def make_report_table(df_month: pd.DataFrame, year: int) -> pd.DataFrame:
    cols = [_month_label(d) for d in df_month["Data"]]

    # Estrutura parecida ao report atual (com separadores e título de ano)
    spec = [
        ("", None, "TITLE"),
        ("Nº clientes com acesso", "Nº clientes com acesso", "INT"),
        ("Nº pedidos pendentes", "Nº pedidos pendentes", "INT"),
        ("Novos pedidos", "Novos pedidos", "INT"),
        ("Desistências (Total)", "Desistências (Total)", "INT"),
        ("De Ativados", "De Ativados", "INT"),
        ("De Pendentes", "De Pendentes", "INT"),
        ("", None, "SEP"),
        ("TOTAL", None, "TITLE"),
        ("Ativados c/ operações (Série 1)", "Ativados c/ operações (Série 1)", "INT"),
        ("% operações / acesso (Série 1)", "% operações / acesso (Série 1)", "PCT"),
        ("", None, "SEP"),
        (str(year), None, "TITLE"),
        ("Ativados c/ operações (Série 2)", "Ativados c/ operações (Série 2)", "INT"),
        ("% operações / acesso (Série 2)", "% operações / acesso (Série 2)", "PCT"),
        ("Nº operações", "Nº operações", "INT"),
        ("Volume negócios", "Volume negócios", "EURC"),
        ("Margem líquida", "Margem líquida", "EURF"),
    ]

    data = []
    idx = []
    row_types = []

    for label, col, kind in spec:
        if kind == "SEP":
            idx.append(" ")
            data.append(["" for _ in cols])
            row_types.append("SEP")
            continue

        if kind == "TITLE":
            idx.append(label if label else " ")
            data.append(["" for _ in cols])
            row_types.append("TITLE")
            continue

        idx.append(label)
        row_types.append("DATA")

        vals = []
        for i in range(len(df_month)):
            v = df_month.iloc[i].get(col) if col else np.nan
            if kind == "INT":
                vals.append(_fmt_int(v))
            elif kind == "PCT":
                vals.append(_fmt_pct(v))
            elif kind == "EURC":
                vals.append(_fmt_eur_compact(v, decimals=1))
            elif kind == "EURF":
                vals.append(_fmt_eur_full(v))
            else:
                vals.append("")
        data.append(vals)

    out = pd.DataFrame(data, index=idx, columns=cols)
    out.attrs["row_types"] = row_types
    return out


def render_report_html(df_report: pd.DataFrame) -> str:
    row_types = df_report.attrs.get("row_types", ["DATA"] * len(df_report))

    # header
    ths = "".join([f"<th>{c}</th>" for c in df_report.columns])

    # rows
    trs = []
    for (idx, row), rtype in zip(df_report.iterrows(), row_types):
        if rtype == "SEP":
            trs.append(f"<tr class='sep-row'><td class='rowhdr'></td>{''.join(['<td></td>' for _ in df_report.columns])}</tr>")
            continue
        if rtype == "TITLE":
            # título ocupa a primeira coluna, resto vazio
            trs.append(
                "<tr class='title-row'>"
                f"<td class='rowhdr'>{idx}</td>"
                f"{''.join(['<td></td>' for _ in df_report.columns])}"
                "</tr>"
            )
            continue

        tds = "".join([f"<td>{row[c]}</td>" for c in df_report.columns])
        trs.append(f"<tr><td class='rowhdr'>{idx}</td>{tds}</tr>")

    body = "\n".join(trs)

    html = f"""
    <div class='report-wrap'>
      <table class='report'>
        <thead>
          <tr>
            <th class='corner'></th>
            {ths}
          </tr>
        </thead>
        <tbody>
          {body}
        </tbody>
      </table>
    </div>
    """
    return html


# =============================================================================
# UI (SEM sidebar)
# =============================================================================

st.title("📊 Plataforma Cambial — Dashboard")
st.markdown("<div class='subtle'>Vista principal: <b>anual</b> (mês a mês). Vista secundária: <b>Hoje</b> (diário) para confirmar estado atual.</div>", unsafe_allow_html=True)

# Upload no corpo
with st.expander("📥 Carregar CSV", expanded=True):
    c1, c2 = st.columns([2, 1])
    with c1:
        upl = st.file_uploader("CSV", type=["csv"], label_visibility="collapsed")
    with c2:
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
    st.info("Carrega o CSV para ver o dashboard.")
    st.stop()

try:
    df_daily = load_report(raw)
except Exception as e:
    st.error(f"Erro ao ler o CSV: {e}")
    st.stop()

if df_daily.empty:
    st.warning("CSV sem dados válidos.")
    st.stop()

# Ano disponível
years = sorted(df_daily["Data"].dt.year.dropna().unique().tolist())
sel_year = years[-1]

cA, cB, cC = st.columns([1, 1, 2])
with cA:
    sel_year = st.selectbox("Ano", years, index=len(years) - 1)
with cB:
    # controlo simples, sem sidebar
    show_charts = st.toggle("Mostrar 3 gráficos mensais", value=True)
with cC:
    st.markdown(f"<div class='subtle'>Última data no ficheiro: <b>{df_daily['Data'].max().date()}</b></div>", unsafe_allow_html=True)

st.divider()

# Tabs (Anual por default)
tab_annual, tab_today = st.tabs(["📅 Anual (mês a mês)", "📆 Hoje (diário)"])

# =============================================================================
# TAB: ANUAL
# =============================================================================

with tab_annual:
    df_month = build_monthly_year(df_daily, int(sel_year))
    if df_month.empty:
        st.warning("Sem dados para o ano selecionado.")
        st.stop()

    st.subheader(f"{sel_year} — Resumo mensal (como o report)")
    st.markdown("<div class='subtle'>Stocks = último dia do mês. Fluxos = soma do mês. Operações/Volume/Margem: detetado automaticamente (acumulado vs soma).</div>", unsafe_allow_html=True)

    report_tbl = make_report_table(df_month, int(sel_year))
    st.markdown(render_report_html(report_tbl), unsafe_allow_html=True)

    if show_charts:
        st.divider()
        st.subheader("Comparação mensal (3 gráficos só)")

        # Para os gráficos, renomear colunas para evitar qualquer underscore / nomes internos (já estão limpos)
        dfc = df_month[["Data", "Volume negócios", "Nº operações", "Margem líquida"]].copy()
        dfc = dfc.rename(columns={"Data": "Mês"})

        c1, c2, c3 = st.columns(3)

        with c1:
            st.caption("Volume negócios")
            st.bar_chart(dfc, x="Mês", y="Volume negócios")

        with c2:
            st.caption("Nº operações")
            st.bar_chart(dfc, x="Mês", y="Nº operações")

        with c3:
            st.caption("Margem líquida")
            st.bar_chart(dfc, x="Mês", y="Margem líquida")

    # Download do report mensal em CSV (opcional)
    st.download_button(
        "⬇️ Download (tabela mensal em CSV)",
        data=report_tbl.reset_index().rename(columns={"index": "Métrica"}).to_csv(index=False).encode("utf-8"),
        file_name=f"report_mensal_{sel_year}.csv",
        mime="text/csv",
    )

# =============================================================================
# TAB: HOJE
# =============================================================================

with tab_today:
    st.subheader("Estado atual (última data do ficheiro)")

    last = df_daily.sort_values("Data").iloc[-1]
    last_date = pd.to_datetime(last["Data"]).date()

    def mini_card(title: str, value: str, note: str = ""):
        st.markdown(
            f"""
            <div class="mini-card">
              <div class="mini-title">{title}</div>
              <div class="mini-value">{value}</div>
              <div class="mini-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='mini-grid'>", unsafe_allow_html=True)
    mini_card("Clientes com acesso", _fmt_int(last.get("Nº clientes com acesso")), f"Data: {last_date}")
    mini_card("Pedidos pendentes", _fmt_int(last.get("Nº pedidos pendentes")))
    mini_card("Nº operações", _fmt_int(last.get("Nº operações")))
    mini_card("Volume negócios", _fmt_eur_compact(last.get("Volume negócios"), decimals=1))
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # últimos 30 dias (só 2 gráficos)
    dmax = df_daily["Data"].max()
    df_30 = df_daily[df_daily["Data"] >= (dmax - pd.Timedelta(days=30))].copy().sort_values("Data")

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Volume (últimos 30 dias)")
        chart = df_30[["Data", "Volume negócios"]].rename(columns={"Data": "Dia"})
        st.line_chart(chart, x="Dia", y="Volume negócios")

    with c2:
        st.caption("Nº operações (últimos 30 dias)")
        chart = df_30[["Data", "Nº operações"]].rename(columns={"Data": "Dia"})
        st.line_chart(chart, x="Dia", y="Nº operações")

    with st.expander("Ver dados diários (últimos 60 dias)", expanded=False):
        df_60 = df_daily[df_daily["Data"] >= (dmax - pd.Timedelta(days=60))].copy().sort_values("Data")
        show = df_60.copy()
        show["Data"] = show["Data"].dt.date
        st.dataframe(show, use_container_width=True, height=420)

st.caption("✅ Sem sidebar. ✅ Sem alertas. ✅ Foco no anual (mês a mês).")
