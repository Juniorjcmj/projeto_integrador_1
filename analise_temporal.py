"""Análise de séries temporais da evasão: diagnóstico, anomalias e previsão (ARIMA e Prophet).

Adapta ao tema de desligamentos de militares a sequência de análise do notebook de vazamento de
credenciais usado anteriormente na disciplina (13 gráficos + validação da base). A granularidade aqui é
MENSAL (os desligamentos são registrados por mês), então:
    diário → mensal · semanal → trimestral · dia da semana → mês do ano · 90 dias → 12 meses.

Cada gráfico é seguido de uma caixa "Como ler este gráfico" gerada a partir dos dados do recorte atual.
"""
from __future__ import annotations

import itertools
import logging
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dados import MOTIVOS, NOMES_POSTOS, ORDEM_POSTOS, TEMAS, serie_mensal
from estilos import (AZUL, BORDER, CONFIG_PLOTLY, COR_CATEGORIA, COR_MOTIVO, ESCALA_SEQ, INK, INK_2, MARINHO,
                     MUTED, NEUTRO, checagens, itens, leitura, secao, tema)

MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
CRITICO, ATENCAO, BOM = "#d03b3b", "#fab219", "#0ca30c"   # paleta de status (sempre com rótulo)
CLARO = "#b7d3f6"
MIN_DECOMP = 24        # 2 ciclos anuais completos
MIN_MODELO = 24        # pontos mínimos para ARIMA/Prophet

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


# ====================================================================== utilidades
def _g(fig):
    st.plotly_chart(fig, width="stretch", config=CONFIG_PLOTLY)


def _md(html: str):
    st.markdown(html, unsafe_allow_html=True)


def _pct(x: float) -> str:
    return f"{x:.0%}"


def _num(x: float, casas: int = 1) -> str:
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _mes(ts) -> str:
    return f"{MESES[ts.month - 1]}/{ts.year}"


def serie_modelo(dfp: pd.DataFrame, hoje: pd.Timestamp) -> tuple[pd.Series, bool]:
    """Série mensal para modelagem; descarta o mês corrente, que está incompleto."""
    m = serie_mensal(dfp)
    s = m.set_index("Periodo")["Qtde"].astype(float)
    s.index = pd.DatetimeIndex(s.index, freq="MS")
    parcial = len(s) > 0 and s.index[-1] >= hoje
    if parcial:
        s = s.iloc[:-1]
    return s, parcial


def metricas(real: np.ndarray, prev: np.ndarray) -> dict:
    real, prev = np.asarray(real, float), np.asarray(prev, float)
    erro = real - prev
    return {"MAE": float(np.mean(np.abs(erro))),
            "RMSE": float(np.sqrt(np.mean(erro ** 2))),
            "MAPE": float(np.mean(np.abs(erro) / (real + 1)) * 100)}   # +1 evita divisão por zero


# ====================================================================== cálculos (cacheados)
@st.cache_data(show_spinner=False)
def calc_anomalias(idx: tuple, vals: tuple) -> pd.DataFrame:
    s = pd.Series(vals, index=pd.DatetimeIndex(idx), name="qtde", dtype=float)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    lim = q3 + 1.5 * (q3 - q1)
    z = (s - s.mean()) / (s.std() or 1)
    d = pd.DataFrame({"qtde": s, "z": z, "iqr": s > lim, "zs": z.abs() > 3})
    d["anomalia"] = d["iqr"] | d["zs"]
    d.attrs["lim_iqr"] = float(lim)
    return d


@st.cache_data(show_spinner=False)
def calc_decomposicao(idx: tuple, vals: tuple) -> dict | None:
    from statsmodels.tsa.seasonal import seasonal_decompose

    s = pd.Series(vals, index=pd.DatetimeIndex(idx, freq="MS"), dtype=float)
    if len(s) < MIN_DECOMP:
        return None
    r = seasonal_decompose(s, model="additive", period=12)
    return {"obs": r.observed, "tend": r.trend, "saz": r.seasonal, "res": r.resid}


@st.cache_data(show_spinner=False)
def calc_estacionariedade(idx: tuple, vals: tuple) -> dict:
    from statsmodels.tsa.stattools import acf, adfuller, pacf

    s = pd.Series(vals, index=pd.DatetimeIndex(idx), dtype=float)

    def adf(x):
        r = adfuller(x.dropna(), autolag="AIC")
        return {"stat": r[0], "p": r[1], "lags": r[2], "nobs": r[3], "crit": r[4], "estac": r[1] < 0.05}

    orig = adf(s)
    dif = adf(s.diff().dropna())
    base = s if orig["estac"] else s.diff().dropna()
    n_lags = max(1, min(24, len(base) // 2 - 1))
    a, a_ci = acf(base, nlags=n_lags, alpha=0.05)
    p, p_ci = pacf(base, nlags=n_lags, alpha=0.05, method="ywm")
    return {"orig": orig, "dif": dif, "usou_dif": not orig["estac"], "n_lags": n_lags,
            "acf": a, "acf_ci": a_ci - a[:, None], "pacf": p, "pacf_ci": p_ci - p[:, None]}


@st.cache_data(show_spinner="Ajustando ARIMA (grade de 18 ordens)…")
def calc_arima(idx: tuple, vals: tuple) -> dict | None:
    from scipy import stats
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import acf

    s = pd.Series(vals, index=pd.DatetimeIndex(idx, freq="MS"), dtype=float)
    if len(s) < MIN_MODELO:
        return None
    n_tr = int(len(s) * 0.8)
    treino, teste = s.iloc[:n_tr], s.iloc[n_tr:]
    grade = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ordem in itertools.product(range(3), range(2), range(3)):
            try:
                r = ARIMA(treino, order=ordem).fit()
                if np.isfinite(r.aic):
                    grade.append((ordem, r.aic, r))
            except Exception:
                continue
    if not grade:
        return None
    grade.sort(key=lambda g: g[1])
    ordem, aic, res = grade[0]
    passos = len(teste) + 12
    prev = res.get_forecast(steps=passos)
    media = prev.predicted_mean.clip(lower=0)
    ic = prev.conf_int(alpha=0.05).clip(lower=0)

    burn = res.loglikelihood_burn
    rz = pd.Series(res.standardized_forecasts_error[0][burn:], index=treino.index[burn:]).dropna()
    (osm, osr), (incl, inter, _) = stats.probplot(rz, dist="norm")
    n_lags = max(1, min(12, len(rz) // 2 - 1))
    racf = acf(rz, nlags=n_lags)
    return {"ordem": ordem, "aic": aic, "top": [(o, a) for o, a, _ in grade[:5]],
            "treino": treino, "teste": teste, "media": media, "ic_inf": ic.iloc[:, 0], "ic_sup": ic.iloc[:, 1],
            "metricas": metricas(teste.values, media.iloc[:len(teste)].values),
            "resid": rz, "qq": (osm, osr, incl, inter), "racf": racf, "racf_lim": 1.96 / np.sqrt(len(rz))}


@st.cache_data(show_spinner="Ajustando Prophet…")
def calc_prophet(idx: tuple, vals: tuple) -> dict | None:
    from prophet import Prophet

    s = pd.Series(vals, index=pd.DatetimeIndex(idx, freq="MS"), dtype=float)
    if len(s) < MIN_MODELO:
        return None
    d = s.rename_axis("ds").reset_index(name="y")
    n_tr = int(len(d) * 0.8)
    treino, teste = d.iloc[:n_tr], d.iloc[n_tr:]
    anos = (d["ds"].max() - d["ds"].min()).days / 365.25
    def ajustar(base: pd.DataFrame, periodos: int):
        m = Prophet(seasonality_mode="additive", yearly_seasonality=anos >= 2, weekly_seasonality=False,
                    daily_seasonality=False, changepoint_prior_scale=0.05, interval_width=0.95)
        m.fit(base)
        fc = m.predict(m.make_future_dataframe(periods=periodos, freq="MS"))
        for c in ("yhat", "yhat_lower", "yhat_upper"):
            fc[c] = fc[c].clip(lower=0)
        cols = ["ds", "yhat", "yhat_lower", "yhat_upper", "trend", "trend_lower", "trend_upper"]
        return m, fc[cols + (["yearly"] if "yearly" in fc.columns else [])]

    # Validação: treina nos 80% iniciais e mede o erro nos 20% finais (gráficos 11 e comparação).
    m, fc = ajustar(treino, len(teste) + 12)
    prev_teste = fc.set_index("ds").loc[teste["ds"], "yhat"].values
    # Projeção: reajusta com a série inteira — o futuro deve usar todos os meses conhecidos (gráfico 13).
    m_full, fc_full = ajustar(d, 12)
    return {"treino": treino, "teste": teste, "fc": fc, "fc_full": fc_full, "anual": anos >= 2, "anos": anos,
            "metricas": metricas(teste["y"].values, prev_teste),
            "mudancas": [pd.Timestamp(c) for c in m_full.changepoints[
                np.abs(np.nanmean(m_full.params["delta"], axis=0)) >= 0.01]]}


# ====================================================================== validação da base
def validar_base(df_all: pd.DataFrame, df: pd.DataFrame, dfp: pd.DataFrame, hoje: pd.Timestamp,
                 ocultar_futuro: bool) -> list[tuple[str, str]]:
    ch = []
    obrig = ["ID_Pessoa", "Periodo", "Posto_Quadro", "Motivo_Principal", *TEMAS.values()]
    falt = [c for c in obrig if c not in df_all.columns]
    ch.append(("bloqueio", f"Colunas ausentes: <b>{falt}</b>.") if falt else ("ok", "Todas as colunas obrigatórias estão presentes."))

    dup = int(df_all["ID_Pessoa"].duplicated().sum())
    ch.append(("bloqueio", f"<b>{dup}</b> ID(s) duplicado(s) — contagens infladas.") if dup
              else ("ok", "Nenhum ID de militar duplicado."))

    fora = set(df_all["Motivo_Principal"].dropna()) - set(MOTIVOS)
    ch.append(("alerta", f"Motivos fora da lista: <b>{sorted(fora)}</b>.") if fora
              else ("ok", "Todos os motivos pertencem à lista fechada."))

    if len(dfp):
        ini, fim = dfp["Periodo"].min(), dfp["Periodo"].max()
        n_meses = (fim.year - ini.year) * 12 + fim.month - ini.month + 1
        vazios = n_meses - dfp["Periodo"].nunique()
        ch.append(("ok" if n_meses >= MIN_DECOMP else "alerta",
                   f"Série de <b>{n_meses} meses</b> ({_mes(ini)} a {_mes(fim)}), {vazios} mês(es) sem registro"
                   + ("." if n_meses >= MIN_DECOMP else
                      f" — menos de {MIN_DECOMP} meses: decomposição, sazonalidade anual e modelos ficam indisponíveis.")))
        if fim >= hoje:
            ch.append(("alerta", f"O mês corrente ({_mes(fim)}) está incompleto: aparece nos gráficos descritivos, "
                                 "mas é <b>excluído dos modelos</b> para não parecer uma queda."))

    n = len(df)
    if n:
        sint = (df["Origem"] == "Sintético").mean()
        ch.append(("alerta" if sint >= 0.5 else "ok",
                   f"<b>{_pct(sint)}</b> do recorte é sintético (Cópula Gaussiana)"
                   + (" — tendências e previsões refletem sobretudo o gerador; confirme com o filtro <b>Reais</b>."
                      if sint >= 0.5 else ".")))
        rec = df[df["Periodo"] >= hoje - pd.DateOffset(months=12)]
        if len(rec) and 0 < sint < 1:
            sint_rec = (rec["Origem"] == "Sintético").mean()
            if sint_rec - sint > 0.05:
                ch.append(("alerta", f"Os sintéticos se concentram no fim da série: <b>{_pct(sint_rec)}</b> dos "
                                     f"últimos 12 meses contra {_pct(sint)} no total — a alta recente é inflada."))

    sem = int(df["Periodo"].isna().sum())
    if sem:
        ch.append(("alerta", f"<b>{sem}</b> registro(s) sem data ficam fora das séries temporais."))
    fut = int((df_all["Periodo"] > hoje).sum())
    if fut:
        ch.append(("alerta" if not ocultar_futuro else "ok",
                   f"<b>{fut}</b> registro(s) com data futura na base — "
                   + ("estão <b>ocultos</b> (Opções avançadas)." if ocultar_futuro else "<b>visíveis</b> e distorcem as séries.")))
    grafias = df_all["Posto_Quadro"].dropna().nunique()
    ch.append(("ok", f"<b>{grafias}</b> grafias distintas de posto/quadro normalizadas para "
                     f"{df_all['Posto'].nunique()} postos."))
    return ch


# ====================================================================== aba "Séries temporais"
def aba_series(df_all, df, dfp, hoje, janela, ocultar_futuro):
    s, parcial = serie_modelo(dfp, hoje)
    mensal = serie_mensal(dfp).set_index("Periodo")["Qtde"]
    n_total = len(dfp)

    # ---------------------------------------------------------------- validação
    with st.container(border=True):
        ch = validar_base(df_all, df, dfp, hoje, ocultar_futuro)
        n_bloq = sum(n == "bloqueio" for n, _ in ch)
        n_alert = sum(n == "alerta" for n, _ in ch)
        _md(secao("Validação da base — leia antes dos gráficos",
                  "Checagens automáticas sobre o recorte atual"
                  + (f" · {n_bloq} bloqueio(s)" if n_bloq else "") + (f" · {n_alert} ressalva(s)" if n_alert else "")))
        _md(checagens(ch))
        _md(leitura(
            "Esta lista roda antes de qualquer gráfico e diz <b>se</b> e <b>como</b> eles podem ser lidos.",
            itens("<b>OK</b>: a condição está atendida.",
                  "<b>ATENÇÃO</b>: a análise roda, mas a interpretação muda — a mensagem diz como.",
                  "<b>BLOQUEIO</b>: a base tem um problema que invalida as contagens; corrija antes de concluir."),
            "O ponto mais importante para este tema é a proporção de registros sintéticos: como eles foram gerados "
            "para ampliar a amostra, qualquer tendência ou previsão precisa ser confirmada no filtro <b>Reais</b>."))

    # ---------------------------------------------------------------- 1. distribuição geral
    with st.container(border=True):
        _md(secao("1 · Distribuição geral dos desligamentos",
                  "Quem saiu e por qual motivo — o retrato estático antes da linha do tempo"))
        c1, c2, c3 = st.columns([1, 1, 1.35], gap="medium")
        cat = dfp["Categoria"].value_counts().reindex(list(COR_CATEGORIA)).dropna()
        f = go.Figure(go.Bar(x=cat.index, y=cat.values, marker_color=[COR_CATEGORIA[c] for c in cat.index],
                             text=cat.values.astype(int), textposition="outside", cliponaxis=False,
                             textfont=dict(color=INK), hovertemplate="%{x}: %{y}<extra></extra>"))
        with c1:
            _md('<div class="sec-s">Desligamentos por categoria</div>')
            _g(tema(f, 300, yaxis=dict(visible=False, range=[0, cat.max() * 1.2]), bargap=0.4))
        mot = dfp["Motivo_Principal"].value_counts().reindex(MOTIVOS).dropna()
        f = go.Figure(go.Pie(labels=mot.index, values=mot.values, hole=0.58, sort=False, direction="clockwise",
                             marker=dict(colors=[COR_MOTIVO[m] for m in mot.index], line=dict(color="#fff", width=2)),
                             textinfo="percent", textfont=dict(size=11),
                             hovertemplate="%{label}: %{value} (%{percent})<extra></extra>"))
        f.add_annotation(text=f"<b>{n_total}</b><br>saídas", showarrow=False, font=dict(size=14, color=INK))
        with c2:
            _md('<div class="sec-s">Motivo principal (% do total)</div>')
            _g(tema(f, 300, showlegend=False, margin=dict(l=8, r=8, t=8, b=8)))
        ct = pd.crosstab(dfp["Categoria"], dfp["Motivo_Principal"]).reindex(list(COR_CATEGORIA)).dropna(how="all")
        ct = ct[[m for m in MOTIVOS if m in ct.columns]]
        nlin = ct.sum(axis=1)
        ctp = ct.div(nlin, axis=0).mul(100)
        ctp.index = [f"{i} (n={int(k)})" for i, k in zip(ctp.index, nlin)]
        ctp.columns = [c.replace(" / Vida Pessoal", "").replace(" ", "<br>") for c in ctp.columns]
        f = go.Figure(go.Heatmap(z=ctp.values, x=ctp.columns, y=ctp.index, colorscale=ESCALA_SEQ, zmin=0, zmax=100,
                                 text=np.round(ctp.values).astype(int), texttemplate="%{text}%", showscale=False,
                                 xgap=2, ygap=2, hovertemplate="%{y} · %{x}: %{z:.0f}%<extra></extra>"))
        with c3:
            _md('<div class="sec-s">Categoria × motivo (% dentro de cada categoria)</div>')
            _g(tema(f, 300, xaxis=dict(side="top", tickangle=0, tickfont=dict(color=INK, size=11)),
                    yaxis=dict(tickfont=dict(color=INK), autorange="reversed")))
        topm = mot.idxmax()
        _md(leitura(
            f"<b>Barras (esquerda)</b> — total de saídas por categoria. Oficiais somam <b>{int(cat.get('Oficial', 0))}</b> "
            f"e praças <b>{int(cat.get('Praça', 0))}</b>. A leitura é de <b>volume</b>, não de risco: sem o efetivo de "
            "cada categoria (o denominador), a barra maior pode refletir apenas um contingente maior.",
            f"<b>Rosca (centro)</b> — a divisão das saídas pelo motivo principal. <b>{topm}</b> responde por "
            f"<b>{_pct(mot.max() / n_total)}</b>. É o indicador mais direto de onde a retenção deveria agir.",
            "<b>Mapa de calor (direita)</b> — cruza as duas variáveis, já <b>normalizado por linha</b>: cada categoria "
            "soma 100%. Assim categorias de tamanhos diferentes ficam comparáveis; com números absolutos, a maior "
            "categoria dominaria o mapa inteiro.",
            "<b>O que procurar:</b> um motivo muito mais forte em uma categoria do que na outra — indica que a "
            "causa da saída é específica daquele grupo, e não um problema geral."))

    # ---------------------------------------------------------------- 2. perfil por posto
    with st.container(border=True):
        h1, h2 = st.columns([2.4, 1])
        tema_sel = h2.selectbox("Indicador", list(TEMAS.values()),
                                index=list(TEMAS.values()).index("Insatisfação Profissional"),
                                label_visibility="collapsed")
        h1.markdown(secao("2 · Perfil por posto/graduação",
                          f"Volume de saídas e % que cita “{tema_sel}” em cada posto"), unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="medium")
        vol = dfp["Posto"].value_counts().reindex(ORDEM_POSTOS).dropna().sort_values()
        f = go.Figure(go.Bar(x=vol.values, y=vol.index, orientation="h", marker_color=AZUL,
                             text=vol.values.astype(int), textposition="outside", cliponaxis=False,
                             textfont=dict(color=INK), customdata=[NOMES_POSTOS.get(p, p) for p in vol.index],
                             hovertemplate="%{customdata}: %{x}<extra></extra>"))
        with c1:
            _md('<div class="sec-s">Desligamentos por posto (volume)</div>')
            _g(tema(f, 320, xaxis=dict(visible=False, range=[0, vol.max() * 1.18]), bargap=0.35,
                    yaxis=dict(tickfont=dict(color=INK))))
        grp = dfp[dfp["Posto"].isin(ORDEM_POSTOS)].groupby("Posto")
        taxa = (grp[tema_sel].mean() * 100).sort_values()
        n_p = grp.size().reindex(taxa.index)
        med, dp = taxa.mean(), taxa.std() if len(taxa) > 1 else 0
        faixa = ["Acima da média + 1 DP" if v > med + dp else "Abaixo da média − 1 DP" if v < med - dp
                 else "Dentro da faixa" for v in taxa]
        cor = {"Acima da média + 1 DP": CRITICO, "Dentro da faixa": ATENCAO, "Abaixo da média − 1 DP": BOM}
        f = go.Figure()
        for rot in cor:
            sel = [i for i, r in enumerate(faixa) if r == rot]
            if sel:
                f.add_bar(x=taxa.values[sel], y=taxa.index[sel], orientation="h", name=rot, marker_color=cor[rot],
                          text=[f"{taxa.values[i]:.0f}% (n={int(n_p.values[i])})" for i in sel],
                          textposition="outside", cliponaxis=False, textfont=dict(color=INK))
        f.add_vline(x=med, line=dict(color=INK, width=1.5, dash="dash"),
                    annotation_text=f"média {med:.0f}%", annotation_position="top", annotation_font_color=INK_2)
        with c2:
            _md(f'<div class="sec-s">% que cita “{tema_sel}” por posto</div>')
            _g(tema(f, 320, barmode="overlay", xaxis=dict(visible=False, range=[0, max(taxa.max(), 1) * 1.45]),
                    yaxis=dict(categoryorder="array", categoryarray=list(taxa.index), tickfont=dict(color=INK)),
                    bargap=0.35, legend=dict(orientation="h", y=-0.08, yanchor="top", x=0)))
        pior = taxa.idxmax()
        _md(leitura(
            "O gráfico desce do nível geral para o <b>posto/graduação</b> — é o que transforma o diagnóstico em "
            "<b>endereço</b>: indica em qual grupo concentrar as ações de retenção.",
            "<b>Esquerda — volume.</b> Ranking de saídas por posto. Mesma ressalva do gráfico 1: um posto com "
            "mais militares naturalmente tem mais saídas.",
            f"<b>Direita — % que cita o indicador.</b> Esta métrica já vem <b>normalizada</b> (dentro de cada posto), "
            f"então postos de tamanhos diferentes são comparáveis. <b>{pior}</b> tem a maior taxa "
            f"({taxa.max():.0f}%). O semáforo é calibrado pela própria distribuição: "
            f"vermelho acima de média + 1 desvio-padrão ({med + dp:.0f}%), verde abaixo de média − 1 DP "
            f"({max(med - dp, 0):.0f}%), amarelo entre os dois. A linha tracejada é a média ({med:.0f}%).",
            "<b>O que procurar:</b> um posto <b>pequeno à esquerda e vermelho à direita</b> — o problema some nos "
            "totais, mas proporcionalmente aquele grupo está mais insatisfeito. Postos com <b>n</b> baixo "
            "(poucos casos) variam muito: confirme antes de concluir. Troque o indicador no seletor acima."))

    # ---------------------------------------------------------------- 3. duas escalas
    with st.container(border=True):
        _md(secao("3 · Visão temporal em duas escalas",
                  f"Mensal (o evento) e trimestral (o movimento) · média móvel de {janela} meses"))
        mm = mensal.rolling(janela, center=True, min_periods=1).mean()
        mediana = float(mensal.median())
        f = go.Figure()
        f.add_bar(x=mensal.index, y=mensal.values, name="Desligamentos no mês", marker_color=CLARO,
                  hovertemplate="%{y}<extra>no mês</extra>")
        f.add_scatter(x=mm.index, y=mm.values, mode="lines", name=f"Média móvel {janela} meses (centrada)",
                      line=dict(color=MARINHO, width=2.5), hovertemplate="%{y:.1f}<extra>média móvel</extra>")
        f.add_hline(y=mediana, line=dict(color=INK_2, width=1.2, dash="dot"),
                    annotation_text=f"mediana {mediana:.0f}/mês", annotation_position="top left",
                    annotation_font_color=INK_2)
        _md('<div class="sec-s">Volume mensal</div>')
        _g(tema(f, 300, hovermode="x unified", bargap=0.15, xaxis_hoverformat="%b/%Y"))

        tri = mensal.resample("QS").sum()
        ult_parcial = len(mensal) and (mensal.index[-1].month % 3 != 0 or mensal.index[-1] >= hoje)
        media_t = float(tri.iloc[:-1].mean() if ult_parcial and len(tri) > 1 else tri.mean())
        rot = [f"{d.year} T{(d.month - 1) // 3 + 1}" for d in tri.index]
        acima = tri.values > media_t
        f = go.Figure()
        for flag, nome, c in ((True, "Acima da média", CRITICO), (False, "Abaixo da média", AZUL)):
            sel = [i for i, a in enumerate(acima) if a == flag]
            f.add_bar(x=[rot[i] for i in sel], y=tri.values[sel], name=nome, marker_color=c,
                      marker_opacity=[0.45 if (ult_parcial and i == len(tri) - 1) else 1 for i in sel],
                      hovertemplate="%{x}: %{y}<extra>" + nome + "</extra>")
        f.add_hline(y=media_t, line=dict(color=INK, width=1.5, dash="dash"),
                    annotation_text=f"média {media_t:.0f}/trimestre", annotation_position="top left")
        _md('<div class="sec-s">Volume trimestral (vermelho = acima da média · azul = abaixo)</div>')
        _g(tema(f, 280, barmode="overlay", xaxis=dict(categoryorder="array", categoryarray=rot), bargap=0.2))
        media_m = float(mensal.mean())
        _md(leitura(
            "O mesmo fenômeno em duas granularidades: a escala <b>mensal</b> mostra cada evento; a "
            "<b>trimestral</b>, o movimento de fundo.",
            "<b>Mensal.</b> As barras claras são o total de cada mês — a variação “serrilhada” é esperada, porque "
            f"desligamentos se concentram em datas específicas. A <b>linha escura</b> é a média móvel centrada de "
            f"{janela} meses: suaviza o ruído e revela a tendência. A <b>linha pontilhada</b> é a mediana "
            f"(<b>{mediana:.0f}</b>/mês) — compare com a média (<b>{_num(media_m)}</b>/mês): quanto maior a distância, "
            "mais poucos meses de pico puxam o total e menos a média representa um mês típico.",
            "<b>Trimestral.</b> Cada barra soma três meses. <b>Vermelho</b> = trimestre acima da média do período; "
            "<b>azul</b> = abaixo; a tracejada é a média"
            + (". A última barra, mais clara, é um <b>trimestre incompleto</b> — barra baixa ali é recorte, não queda."
               if ult_parcial else "."),
            "<b>O que procurar:</b> <b>sequências de barras vermelhas</b> indicam piora sustentada, diferente de um "
            "pico isolado. Como a média é puxada pelos picos, “abaixo da média” é um trimestre comum, não "
            "necessariamente bom."))

    # ---------------------------------------------------------------- 4. por motivo
    with st.container(border=True):
        _md(secao("4 · Séries por motivo principal",
                  f"Média móvel de {janela} meses — o movimento geral é puxado por um único motivo?"))
        grupos = dfp.assign(G=dfp["Motivo_Principal"].where(
            dfp["Motivo_Principal"].isin(["Concurso Público", "Família / Vida Pessoal"]), "Demais motivos"))
        sm = serie_mensal(grupos, "G").pivot(index="Periodo", columns="G", values="Qtde").fillna(0)
        cores = {**COR_MOTIVO, "Demais motivos": NEUTRO}
        f = go.Figure()
        for g in ["Concurso Público", "Família / Vida Pessoal", "Demais motivos"]:
            if g in sm:
                v = sm[g].rolling(janela, center=True, min_periods=1).mean()
                f.add_scatter(x=v.index, y=v.values, mode="lines", name=g, line=dict(color=cores[g], width=2.5),
                              hovertemplate="%{y:.1f}<extra>" + g + "</extra>")
        _g(tema(f, 320, hovermode="x unified", xaxis_hoverformat="%b/%Y",
                yaxis=dict(title=dict(text="saídas/mês (suavizado)", font=dict(color=MUTED, size=11)))))
        _md(leitura(
            "Enquanto o gráfico 3 mostra o total, este <b>separa a série por motivo</b> para responder: a alta ou "
            "a queda é geral ou vem de um motivo específico?",
            f"<b>Atenção ao eixo:</b> os valores são a <b>média móvel de {janela} meses</b>, não a contagem real do mês. "
            "A suavização é necessária porque, ao dividir a série, cada linha fica com poucos casos por mês e o "
            "gráfico bruto viraria um emaranhado de picos. Use-o para comparar <b>formato e momento</b> das curvas, "
            "não para ler valores exatos.",
            itens("<b>Picos simultâneos</b> em todas as linhas → um fator comum (ex.: fim de ciclo, calendário de "
                  "concursos e movimentações) afeta todos os motivos ao mesmo tempo.",
                  "<b>Pico em uma só linha</b> → evento específico daquele motivo (ex.: um grande concurso público).",
                  "<b>Cruzamento de linhas</b> → mudança no motivo dominante ao longo do tempo.",
                  "<b>Linha baixa e plana</b> → motivo raro no período — ou sub-registrado nas entrevistas.")))

    # ---------------------------------------------------------------- 5. sazonalidade anual
    with st.container(border=True):
        _md(secao("5 · Sazonalidade ao longo do ano",
                  "Os meses são “dobrados”: todos os janeiros ficam juntos, todos os fevereiros juntos…"))
        c1, c2 = st.columns(2, gap="medium")
        por_mes = mensal.groupby(mensal.index.month).sum().reindex(range(1, 13), fill_value=0)
        f = go.Figure(go.Bar(x=MESES, y=por_mes.values,
                             marker_color=[MARINHO if v == por_mes.max() else CLARO for v in por_mes.values],
                             text=por_mes.values, textposition="outside", cliponaxis=False,
                             textfont=dict(color=INK), hovertemplate="%{x}: %{y}<extra>total</extra>"))
        f.add_hline(y=por_mes.mean(), line=dict(color=INK, width=1.2, dash="dash"),
                    annotation_text=f"média {por_mes.mean():.0f}", annotation_position="top left")
        with c1:
            _md('<div class="sec-s">Total acumulado por mês do ano</div>')
            _g(tema(f, 300, yaxis=dict(visible=False, range=[0, por_mes.max() * 1.2]), bargap=0.25))
        f = go.Figure()
        for i, nome in enumerate(MESES, 1):
            v = mensal[mensal.index.month == i]
            f.add_box(y=v.values, name=nome, marker_color=AZUL, line=dict(width=1.3), fillcolor="rgba(42,120,214,.25)",
                      boxpoints="outliers", marker=dict(color=CRITICO, size=6), showlegend=False,
                      hovertemplate="%{y}<extra>" + nome + "</extra>")
        with c2:
            _md('<div class="sec-s">Distribuição dos meses (boxplot)</div>')
            _g(tema(f, 300))
        mx = MESES[int(por_mes.idxmax()) - 1]
        n_anos = mensal.index.year.nunique()
        _md(leitura(
            "Sazonalidade é o padrão que <b>se repete em ciclo fixo</b> — aqui, o ano. O eixo horizontal deixa de "
            "ser linha do tempo e passa a ser a posição do mês dentro do ano.",
            f"<b>Esquerda — total por mês.</b> Soma de todas as saídas de cada mês do calendário ao longo dos "
            f"{n_anos} anos da série. <b>{mx}</b> (destacado) acumula mais saídas; a tracejada é a média entre os meses. "
            "Responde: <i>em que mês saem mais militares?</i>",
            "<b>Direita — boxplot.</b> Responde a uma pergunta mais útil: <i>o mês é consistentemente alto ou um único "
            "ano inflou o total?</i>",
            itens("A <b>linha dentro da caixa</b> é a mediana — o valor típico daquele mês.",
                  "A <b>caixa</b> vai do 1º ao 3º quartil (metade dos anos cai nela); caixa estreita = mês previsível.",
                  "Os <b>bigodes</b> vão até 1,5 × o intervalo entre quartis.",
                  "<b>Pontos vermelhos</b> são anos fora do padrão daquele mês — devem coincidir com as anomalias do gráfico 7."),
            "<b>A comparação que importa:</b> mês com total alto à esquerda mas mediana baixa e um ponto isolado à "
            "direita não tem padrão — teve um evento único. Mediana alta com caixa estreita indica calendário "
            f"recorrente (ex.: datas de posse de concursos, fim de compromisso de serviço). "
            f"<span class='alerta'>Com {n_anos} anos, cada caixa tem no máximo {n_anos} pontos: trate como indício.</span>"))

    # ---------------------------------------------------------------- 6. decomposição
    with st.container(border=True):
        _md(secao("6 · Decomposição da série: tendência + sazonalidade + resíduo",
                  "Modelo aditivo com ciclo de 12 meses"))
        dec = calc_decomposicao(tuple(s.index), tuple(s.values)) if len(s) else None
        if dec is None:
            st.info(f"A decomposição exige pelo menos {MIN_DECOMP} meses completos (2 ciclos anuais); o recorte tem {len(s)}.")
        else:
            f = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                              subplot_titles=("Série original", "Tendência", "Sazonalidade", "Resíduo"))
            f.add_scatter(x=dec["obs"].index, y=dec["obs"], mode="lines", line=dict(color=AZUL, width=1.8), row=1, col=1)
            f.add_scatter(x=dec["tend"].index, y=dec["tend"], mode="lines", line=dict(color=MARINHO, width=2.5), row=2, col=1)
            f.add_scatter(x=dec["saz"].index, y=dec["saz"], mode="lines", line=dict(color="#1baf7a", width=1.8), row=3, col=1)
            r = dec["res"]
            f.add_bar(x=r.index, y=r.values, marker_color=[CRITICO if v > 0 else AZUL for v in r.fillna(0)], row=4, col=1)
            f.add_hline(y=0, line=dict(color=INK, width=0.8), row=4, col=1)
            tema(f, 620, showlegend=False, hovermode="x unified", xaxis4_hoverformat="%b/%Y")
            for i in range(1, 5):
                f.update_yaxes(gridcolor="#eef0f3", zeroline=False, tickfont=dict(color=MUTED), row=i, col=1)
            f.update_annotations(font=dict(size=12, color=INK_2), x=0, xanchor="left")
            _g(f)
            t = dec["tend"].dropna()
            sentido = "subiu" if t.iloc[-1] > t.iloc[0] else "caiu"
            saz = dec["saz"].iloc[:12]
            pico_s, vale_s = MESES[saz.idxmax().month - 1], MESES[saz.idxmin().month - 1]
            maior_res = dec["res"].abs().idxmax()
            _md(leitura(
                "Uma “cirurgia” na série: separa o <b>rumo de longo prazo</b>, o <b>ciclo que se repete</b> e o "
                "<b>imprevisto</b>. No modelo aditivo, <b>Série = Tendência + Sazonalidade + Resíduo</b>: somando os "
                "três painéis de baixo, reconstrói-se o de cima. Todos compartilham o eixo do tempo.",
                itens(f"<b>Tendência</b> — o movimento de fundo, sem ciclo nem ruído. Responde <i>estamos piorando?</i> "
                      f"No recorte, ela <b>{sentido}</b> de {_num(t.iloc[0])} para {_num(t.iloc[-1])} saídas/mês. "
                      "Começa e termina cortada: a média interna precisa de 6 meses de cada lado.",
                      f"<b>Sazonalidade</b> — o padrão repetido a cada 12 meses, na mesma unidade da série "
                      f"(saídas/mês): <b>{pico_s}</b> fica acima da tendência e <b>{vale_s}</b>, abaixo. O desenho é "
                      "idêntico em todos os anos — isso é característica do método, não descoberta nos dados.",
                      "<b>Resíduo</b> — o que sobra. <b>Vermelho</b> = mais saídas do que tendência e sazonalidade "
                      "explicam; <b>azul</b> = menos."),
                f"<b>O que procurar:</b> o resíduo ideal é pequeno e aleatório em torno de zero. Barras grandes e "
                f"isoladas são os meses que nada explica — aqui, o maior é <b>{_mes(maior_res)}</b>. São os candidatos a "
                "eventos reais (um concurso, uma mudança de política). Resíduo com ondas ou amplitude crescente "
                "indica que o modelo aditivo não captura bem a estrutura."))

    # ---------------------------------------------------------------- 7. anomalias
    with st.container(border=True):
        _md(secao("7 · Detecção de anomalias", "Meses com volume estatisticamente fora do padrão (IQR e Z-Score)"))
        if len(mensal) < 6:
            st.info("São necessários pelo menos 6 meses para detectar anomalias.")
        else:
            an = calc_anomalias(tuple(mensal.index), tuple(mensal.values.astype(float)))
            lim = an.attrs["lim_iqr"]
            a = an[an["anomalia"]]
            f = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.62, 0.38],
                              subplot_titles=("Série mensal com anomalias", "Z-Score (desvios-padrão da média)"))
            f.add_bar(x=an.index, y=an["qtde"], marker_color=CLARO, name="Saídas no mês", row=1, col=1)
            mm7 = an["qtde"].rolling(janela, center=True, min_periods=1).mean()
            f.add_scatter(x=mm7.index, y=mm7, mode="lines", name=f"Média móvel {janela} meses",
                          line=dict(color="#1baf7a", width=2), row=1, col=1)
            f.add_scatter(x=a.index, y=a["qtde"], mode="markers", name=f"Anomalias ({len(a)} meses)",
                          marker=dict(symbol="triangle-down", size=13, color=CRITICO, line=dict(color="#fff", width=1.5)),
                          row=1, col=1)
            f.add_hline(y=lim, line=dict(color=ATENCAO, width=1.8, dash="dash"), row=1, col=1,
                        annotation_text=f"limiar IQR {lim:.0f}", annotation_position="top left")
            f.add_bar(x=an.index, y=an["z"], marker_color=[CRITICO if v > 0 else AZUL for v in an["z"]],
                      name="Z-Score", showlegend=False, row=2, col=1)
            for v in (3, -3):
                f.add_hline(y=v, line=dict(color=CRITICO if v > 0 else AZUL, width=1.4, dash="dash"), row=2, col=1)
            tema(f, 520, hovermode="x unified", barmode="overlay", xaxis2_hoverformat="%b/%Y")
            f.update_annotations(font=dict(size=12, color=INK_2))
            for i in (1, 2):
                f.update_yaxes(gridcolor="#eef0f3", zeroline=False, tickfont=dict(color=MUTED), row=i, col=1)
            _g(f)
            lista = ", ".join(f"{_mes(i)} ({int(v)})" for i, v in a["qtde"].nlargest(6).items()) or "nenhum"
            _md(leitura(
                "Aqui a análise deixa de descrever e passa a <b>sinalizar</b> meses que merecem investigação "
                "individual. Dois métodos rodam juntos e o alerta dispara se <b>qualquer um</b> acusar.",
                f"<b>Painel superior.</b> Barras claras = saídas do mês; linha verde = média móvel (o “normal”); "
                f"<b>triângulos vermelhos</b> = meses anômalos; tracejada amarela = <b>limiar IQR</b> "
                f"(3º quartil + 1,5 × intervalo entre quartis = <b>{lim:.0f}</b> saídas). O IQR é robusto: os quartis "
                "não são puxados pelos próprios picos que se quer detectar.",
                "<b>Painel inferior.</b> Cada mês em <b>desvios-padrão de distância da média</b> (z). As tracejadas em "
                "±3 são o segundo critério — numa distribuição normal, só ~0,3% dos pontos passariam delas.",
                f"<b>Meses sinalizados:</b> {lista}.",
                "<span class='alerta'><b>Limitação:</b> o Z-Score usa média e desvio calculados <i>incluindo</i> os "
                "picos; um pico muito grande infla o desvio e esconde picos menores. Por isso o IQR roda junto — se "
                "discordarem, confie mais no IQR.</span>",
                itens("<b>Anomalia isolada</b> → evento pontual (ex.: posse de um concurso grande).",
                      "<b>Anomalias em meses seguidos</b> → processo em andamento — o padrão mais preocupante.",
                      "Cada triângulo é uma data a cruzar com eventos conhecidos: editais, reestruturações, "
                      "mudanças de carreira ou remuneração.")))


# ====================================================================== aba "Previsão"
def aba_previsao(dfp, hoje):
    s, parcial = serie_modelo(dfp, hoje)
    idx, vals = tuple(s.index), tuple(s.values)
    if len(s) < MIN_MODELO:
        st.info(f"Os modelos precisam de pelo menos {MIN_MODELO} meses completos; o recorte atual tem {len(s)}. "
                "Amplie o período ou a origem dos registros na barra lateral.")
        return
    _md('<div class="nota" style="margin-bottom:.8rem">Série usada nos modelos: '
        f"{len(s)} meses completos ({_mes(s.index[0])} a {_mes(s.index[-1])})"
        + (" — o mês corrente, incompleto, foi excluído." if parcial else ".") + "</div>")

    # ---------------------------------------------------------------- 8. estacionariedade
    with st.container(border=True):
        _md(secao("8 · Teste de estacionariedade e ACF/PACF",
                  "Diagnóstico técnico para escolher os parâmetros do ARIMA"))
        e = calc_estacionariedade(idx, vals)
        tab = pd.DataFrame([
            {"Série": "Original", "Estatística ADF": e["orig"]["stat"], "p-valor": e["orig"]["p"],
             "Conclusão": "Estacionária" if e["orig"]["estac"] else "Não estacionária"},
            {"Série": "1ª diferença", "Estatística ADF": e["dif"]["stat"], "p-valor": e["dif"]["p"],
             "Conclusão": "Estacionária" if e["dif"]["estac"] else "Não estacionária"}])
        st.dataframe(tab, hide_index=True, width="stretch",
                     column_config={"Estatística ADF": st.column_config.NumberColumn(format="%.3f"),
                                    "p-valor": st.column_config.NumberColumn(format="%.4f")})
        c1, c2 = st.columns(2, gap="medium")
        for col, chave, nome, cor in ((c1, "acf", "Autocorrelação (ACF) → sugere q", AZUL),
                                      (c2, "pacf", "Autocorrelação parcial (PACF) → sugere p", CRITICO)):
            v, ci = e[chave], e[chave + "_ci"]
            lags = np.arange(len(v))
            f = go.Figure()
            f.add_scatter(x=np.r_[lags, lags[::-1]], y=np.r_[ci[:, 1], ci[::-1, 0]], fill="toself",
                          fillcolor="rgba(42,120,214,.12)", line=dict(width=0), hoverinfo="skip", name="IC 95%")
            f.add_bar(x=lags, y=v, marker_color=cor, width=0.35, name="correlação",
                      hovertemplate="lag %{x}: %{y:.2f}<extra></extra>")
            f.add_hline(y=0, line=dict(color=INK, width=0.8))
            with col:
                _md(f'<div class="sec-s">{nome}</div>')
                _g(tema(f, 280, showlegend=False, yaxis=dict(range=[-1.05, 1.05]),
                        xaxis=dict(title=dict(text="defasagem (meses)", font=dict(color=MUTED, size=11)))))
        _md(leitura(
            "Ferramentas de <b>diagnóstico técnico</b>, não de comunicação executiva: servem para escolher os "
            "parâmetros do ARIMA.",
            f"<b>Tabela — teste ADF (Dickey-Fuller aumentado).</b> Uma série é <b>estacionária</b> quando média e "
            f"variância não mudam com o tempo — requisito do ARIMA. p-valor abaixo de 0,05 = estacionária. Aqui a "
            f"série original deu p = {e['orig']['p']:.3f} ({'estacionária' if e['orig']['estac'] else 'não estacionária'})"
            + ("; por isso os gráficos usam a <b>1ª diferença</b> (variação de um mês para o outro)." if e["usou_dif"]
               else "; os gráficos usam a série original."),
            f"<b>Gráficos.</b> O eixo horizontal são as <b>defasagens</b> (1 mês atrás, 2 meses atrás… até "
            f"{e['n_lags']}); o vertical, a correlação (−1 a +1). A <b>faixa azul</b> é o intervalo de confiança de 95%: "
            "barras dentro dela são ruído; só as que <b>ultrapassam</b> a faixa são sinal. O lag 0 vale sempre 1 e "
            "deve ser ignorado.",
            itens("<b>ACF</b> mede a correlação da série com ela mesma defasada, incluindo efeitos indiretos. "
                  "Onde ela corta para dentro da faixa sugere o parâmetro <b>q</b> (média móvel).",
                  "<b>PACF</b> mede o efeito <b>direto</b> de cada defasagem, removendo as intermediárias. "
                  "Onde ela corta sugere <b>p</b> (autorregressivo).",
                  "<b>Barra alta isolada no lag 12</b> → sazonalidade anual não tratada; considerar SARIMA.")))

    # ---------------------------------------------------------------- 9. ARIMA
    ar = calc_arima(idx, vals)
    with st.container(border=True):
        if ar is None:
            st.info("Nenhuma ordem do ARIMA convergiu para este recorte.")
        else:
            o = ar["ordem"]
            _md(secao(f"9 · Previsão ARIMA{o} — desligamentos mensais",
                      f"Ordem escolhida pelo menor AIC entre 18 combinações · treino = 80% iniciais"))
            f = go.Figure()
            f.add_scatter(x=ar["ic_sup"].index, y=ar["ic_sup"], mode="lines", line=dict(width=0),
                          showlegend=False, hoverinfo="skip")
            f.add_scatter(x=ar["ic_inf"].index, y=ar["ic_inf"], mode="lines", line=dict(width=0), fill="tonexty",
                          fillcolor="rgba(208,59,59,.13)", name="IC 95%", hoverinfo="skip")
            f.add_scatter(x=ar["treino"].index, y=ar["treino"], mode="lines", name="Treino",
                          line=dict(color=AZUL, width=2))
            f.add_scatter(x=ar["teste"].index, y=ar["teste"], mode="lines+markers", name="Real (teste)",
                          line=dict(color="#1baf7a", width=2), marker=dict(size=7))
            f.add_scatter(x=ar["media"].index, y=ar["media"], mode="lines", name="Previsão ARIMA",
                          line=dict(color=CRITICO, width=2.2, dash="dash"))
            f.add_vline(x=ar["treino"].index[-1], line=dict(color=MUTED, width=1.4, dash="dot"))
            _g(tema(f, 360, hovermode="x unified", xaxis_hoverformat="%b/%Y"))
            m = ar["metricas"]
            _md(_cartoes_metricas(m))
            top = " · ".join(f"ARIMA{t} AIC {a:.1f}" for t, a in ar["top"])
            _md(leitura(
                f"Primeiro modelo preditivo. A ordem <b>(p, d, q) = {o}</b> não foi fixada: foram testadas 18 "
                f"combinações e escolhida a de menor AIC (critério que equilibra ajuste e simplicidade). "
                f"Cinco melhores: {top}. <b>p</b> = termos autorregressivos, <b>d</b> = diferenciações para "
                "estacionar a série, <b>q</b> = termos de média móvel.",
                itens("<b>Azul (treino)</b> — os 80% iniciais: tudo o que o modelo viu.",
                      "<b>Verde (real / teste)</b> — os 20% finais, escondidos do modelo de propósito.",
                      "<b>Vermelho tracejado</b> — o que o modelo projetou para o teste e para os 12 meses seguintes.",
                      "<b>Faixa vermelha</b> — intervalo de 95%, cortado em zero (não existe saída negativa).",
                      "<b>Linha pontilhada</b> — fronteira entre o que o modelo viu e o que ele projetou."),
                "<b>Como avaliar com honestidade:</b> compare o vermelho com o <b>verde</b>, nunca com o azul — acertar "
                "o treino é trivial. Um modelo utilizável mantém o verde <b>dentro</b> da faixa. A projeção tende a "
                "achatar e a faixa <b>se abre em funil</b>: a incerteza se acumula a cada mês, então os primeiros 2 ou 3 "
                "meses são utilizáveis e o 12º é só ordem de grandeza.",
                f"<b>Métricas no teste:</b> MAE = erro médio em saídas/mês; RMSE = idem, penalizando erros grandes "
                f"(muito acima do MAE = errou feio em meses específicos); MAPE = erro percentual (abaixo de 20% é bom; "
                f"acima de 50%, sem valor preditivo). Aqui o MAPE foi <b>{m['MAPE']:.0f}%</b>."
                + (" <span class='alerta'>Desligamentos são dirigidos por eventos (resultados de concursos, "
                   "movimentações), que o histórico recente não antecipa — o valor da análise está nos gráficos 6 e 7, "
                   "não na previsão.</span>" if m["MAPE"] > 50 else "")))

    # ---------------------------------------------------------------- 10. diagnóstico
    if ar is not None:
        with st.container(border=True):
            _md(secao(f"10 · Diagnóstico dos resíduos do ARIMA{ar['ordem']}",
                      "O modelo capturou toda a estrutura dos dados?"))
            rz = ar["resid"]
            osm, osr, incl, inter = ar["qq"]
            f = make_subplots(rows=2, cols=2, vertical_spacing=0.16, horizontal_spacing=0.08,
                              subplot_titles=("Resíduos padronizados no tempo", "Histograma vs. Normal(0,1)",
                                              "Q-Q plot (normalidade)", "Correlograma dos resíduos"))
            f.add_scatter(x=rz.index, y=rz, mode="lines+markers", line=dict(color=AZUL, width=1.3),
                          marker=dict(size=4), row=1, col=1)
            f.add_hline(y=0, line=dict(color=INK, width=0.8), row=1, col=1)
            f.add_histogram(x=rz, histnorm="probability density", marker_color=CLARO, nbinsx=15, row=1, col=2)
            xs = np.linspace(min(-3.5, rz.min()), max(3.5, rz.max()), 120)
            f.add_scatter(x=xs, y=np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), mode="lines",
                          line=dict(color=CRITICO, width=2), row=1, col=2)
            f.add_scatter(x=osm, y=osr, mode="markers", marker=dict(color=AZUL, size=6), row=2, col=1)
            f.add_scatter(x=[osm.min(), osm.max()], y=[inter + incl * osm.min(), inter + incl * osm.max()],
                          mode="lines", line=dict(color=CRITICO, width=2), row=2, col=1)
            lag = np.arange(1, len(ar["racf"]))
            f.add_bar(x=lag, y=ar["racf"][1:], marker_color=AZUL, width=0.35, row=2, col=2)
            for v in (ar["racf_lim"], -ar["racf_lim"]):
                f.add_hline(y=v, line=dict(color=CRITICO, width=1.2, dash="dash"), row=2, col=2)
            tema(f, 560, showlegend=False)
            f.update_annotations(font=dict(size=12, color=INK_2))
            for r_ in (1, 2):
                for c_ in (1, 2):
                    f.update_yaxes(gridcolor="#eef0f3", zeroline=False, tickfont=dict(color=MUTED), row=r_, col=c_)
            _g(f)
            fora = int(np.sum(np.abs(ar["racf"][1:]) > ar["racf_lim"]))
            _md(leitura(
                "Este painel não avalia a previsão — avalia se o <b>modelo é estatisticamente válido</b>. Se o ARIMA "
                "capturou toda a estrutura, o que sobra (o resíduo) é ruído puro. Padrão no resíduo = informação "
                "que o modelo deixou passar.",
                itens("<b>Resíduos no tempo (↖)</b> — devem formar uma faixa aleatória em torno de zero, de largura "
                      "constante. Sinal ruim: amplitude crescendo ou blocos seguidos só positivos/negativos.",
                      "<b>Histograma (↗)</b> — distribuição dos resíduos contra a curva normal (vermelha). Devem se "
                      "sobrepor. Sinal ruim: assimetria forte ou duas “corcovas” (dois regimes misturados).",
                      "<b>Q-Q plot (↙)</b> — os pontos devem cair sobre a reta. Pontos que fogem <b>nas duas pontas</b> "
                      "(formato de “S”) = caudas pesadas: meses extremos mais frequentes que o previsto. Nesse caso "
                      "o intervalo de 95% do gráfico 9 é <b>otimista</b>.",
                      f"<b>Correlograma (↘)</b> — o painel decisivo. Todas as barras devem ficar entre as tracejadas. "
                      f"Aqui, <b>{fora}</b> barra(s) escapa(m)"
                      + (": sobrou estrutura temporal que a ordem escolhida não modelou (ajustar p/q ou usar SARIMA)."
                         if fora else ": o modelo não deixou autocorrelação relevante para trás.")),
                "<b>Veredito rápido:</b> resíduos aleatórios + Q-Q próximo da reta + correlograma limpo = modelo "
                "confiável. Falhas aqui não invalidam a previsão, mas obrigam a tratar a faixa de confiança como "
                "estreita demais."))

    # ---------------------------------------------------------------- 11. Prophet
    pr = calc_prophet(idx, vals)
    with st.container(border=True):
        _md(secao("11 · Previsão Prophet — desligamentos mensais (12 meses à frente)",
                  "Tendência + sazonalidade anual ajustadas separadamente · treino = 80% iniciais"))
        fc = pr["fc"]
        f = go.Figure()
        f.add_scatter(x=fc["ds"], y=fc["yhat_upper"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        f.add_scatter(x=fc["ds"], y=fc["yhat_lower"], mode="lines", line=dict(width=0), fill="tonexty",
                      fillcolor="rgba(208,59,59,.13)", name="IC 95%", hoverinfo="skip")
        f.add_scatter(x=pr["treino"]["ds"], y=pr["treino"]["y"], mode="markers", name="Treino",
                      marker=dict(color=AZUL, size=6, opacity=0.7))
        f.add_scatter(x=pr["teste"]["ds"], y=pr["teste"]["y"], mode="markers", name="Real (teste)",
                      marker=dict(color="#1baf7a", size=8))
        f.add_scatter(x=fc["ds"], y=fc["yhat"], mode="lines", name="Previsão Prophet",
                      line=dict(color=CRITICO, width=2))
        f.add_vline(x=pr["treino"]["ds"].max(), line=dict(color=MUTED, width=1.4, dash="dot"))
        _g(tema(f, 360, hovermode="x unified", xaxis_hoverformat="%b/%Y"))
        m = pr["metricas"]
        _md(_cartoes_metricas(m))
        teste_prev = fc.set_index("ds").loc[pr["teste"]["ds"], "yhat"].values
        acima = np.mean(pr["teste"]["y"].values > teste_prev)
        _md(leitura(
            "Segundo modelo, com filosofia oposta ao ARIMA: em vez de aprender com a correlação entre meses "
            "vizinhos, o <b>Prophet</b> (Meta) decompõe explicitamente a série em <b>tendência + sazonalidade anual</b> "
            "e ajusta cada parte separadamente. É robusto a picos e a meses faltantes.",
            itens("<b>Pontos azuis</b> — treino (80% iniciais); <b>pontos verdes</b> — o real do período de teste.",
                  "<b>Linha vermelha</b> — cobre todo o período: sobre os pontos azuis é <b>ajuste</b>, não previsão; "
                  "só depois da linha pontilhada é projeção verdadeira, até 12 meses além do último dado.",
                  "<b>Faixa</b> — intervalo de 95%, cortado em zero."),
            "<b>Configuração que muda a leitura:</b> sazonalidade <b>aditiva</b> (a oscilação anual tem tamanho fixo em "
            "saídas/mês) — escolhida porque há meses com zero saídas, o que torna o modo multiplicativo instável; e "
            + ("sazonalidade anual <b>ligada</b>, porque a série cobre "
               f"{_num(pr['anos'])} anos (≥ 2 ciclos)." if pr["anual"] else
               f"sazonalidade anual <b>desligada</b>: a série cobre só {_num(pr['anos'])} anos (&lt; 2 ciclos).")
            + " A tendência muda de direção de forma suave (valor conservador padrão).",
            f"<b>O que procurar:</b> pontos verdes sistematicamente <b>acima</b> da linha = o modelo subestima e o "
            f"cenário real é pior. Aqui, {_pct(acima)} dos meses de teste ficaram acima. MAPE do Prophet: "
            f"<b>{m['MAPE']:.0f}%</b> — comparável ao do ARIMA, pois os dois preveem totais mensais."))

    # ---------------------------------------------------------------- 12. componentes
    with st.container(border=True):
        _md(secao("12 · Componentes do Prophet",
                  "As peças que, somadas, formam a previsão — modelo reajustado com a série inteira"))
        fc = pr["fc_full"]
        tem_anual = pr["anual"] and "yearly" in fc
        f = make_subplots(rows=2 if tem_anual else 1, cols=1, vertical_spacing=0.14,
                          subplot_titles=("Tendência", "Sazonalidade anual (saídas/mês acima ou abaixo da tendência)")
                          if tem_anual else ("Tendência",))
        f.add_scatter(x=fc["ds"], y=fc["trend_upper"], mode="lines", line=dict(width=0), hoverinfo="skip", row=1, col=1)
        f.add_scatter(x=fc["ds"], y=fc["trend_lower"], mode="lines", line=dict(width=0), fill="tonexty",
                      fillcolor="rgba(11,59,117,.12)", hoverinfo="skip", row=1, col=1)
        f.add_scatter(x=fc["ds"], y=fc["trend"], mode="lines", line=dict(color=MARINHO, width=2.5),
                      hovertemplate="%{y:.1f}<extra>tendência</extra>", row=1, col=1)
        for c in pr["mudancas"]:
            f.add_vline(x=c, line=dict(color=MUTED, width=1, dash="dot"), row=1, col=1)
        if tem_anual:
            ano = fc.set_index("ds")["yearly"]
            ano = ano.groupby(ano.index.month).mean().reindex(range(1, 13))
            f.add_bar(x=MESES, y=ano.values, marker_color=[CRITICO if v > 0 else AZUL for v in ano.values],
                      hovertemplate="%{x}: %{y:+.1f}<extra></extra>", row=2, col=1)
            f.add_hline(y=0, line=dict(color=INK, width=0.8), row=2, col=1)
        tema(f, 480 if tem_anual else 300, showlegend=False)
        f.update_annotations(font=dict(size=12, color=INK_2), x=0, xanchor="left")
        _g(f)
        tr = fc["trend"]
        pico_a = MESES[int(np.nanargmax(ano.values))] if tem_anual else None
        _md(leitura(
            "Em termos analíticos, o gráfico mais informativo da previsão: abre a “caixa-preta” e mostra as peças "
            "que, <b>somadas</b> (modo aditivo), produzem a previsão. Diferente do gráfico 11 (treinado só com 80% "
            "dos meses, para medir o erro), aqui o modelo foi <b>reajustado com a série inteira</b> — é ele que gera a "
            "projeção dos próximos 12 meses no painel resumo.",
            f"<b>Tendência</b> — o movimento estrutural, livre de sazonalidade e ruído. Responde à pergunta executiva: "
            f"<i>a evasão está aumentando ou diminuindo?</i> Ela vai de {_num(tr.iloc[0])} a {_num(tr.iloc[-1])} "
            f"saídas/mês no horizonte projetado. As <b>linhas pontilhadas</b> marcam os pontos de mudança de "
            f"inclinação detectados ({len(pr['mudancas'])}) — cada um merece ser cruzado com mudanças de política de "
            "carreira, remuneração ou calendário de concursos naquela data. A faixa é a incerteza da própria tendência.",
            (f"<b>Sazonalidade anual</b> — o efeito médio de cada mês, <b>já isolado da tendência</b>, em saídas/mês: "
             f"barras vermelhas acima de zero = meses com mais saídas que o nível base (o maior é <b>{pico_a}</b>); azuis "
             "= menos. É o mesmo fenômeno do gráfico 5, sem o risco de confundir “este mês é pior” com “o ano em que "
             "houve mais desse mês foi pior”." if tem_anual else
             "<b>Sazonalidade anual</b> — não aparece porque a série tem menos de 2 ciclos completos: com um só ano, "
             "não há como distinguir sazonalidade de tendência, e desligar o componente é a decisão correta."),
            "<b>O que procurar:</b> tendência em alta com sazonalidade forte indica um processo <b>recorrente e "
            "previsível</b> (ex.: calendário anual de concursos) — o que muda a resposta de “investigar casos” para "
            "“agir antes do pico”."))

    # ---------------------------------------------------------------- comparação
    if ar is not None:
        with st.container(border=True):
            _md(secao("Comparação dos modelos", "Erros no mesmo período de teste — menor é melhor"))
            cmp = pd.DataFrame([{"Modelo": f"ARIMA{ar['ordem']}", **ar["metricas"]},
                                {"Modelo": "Prophet", **pr["metricas"]}])
            st.dataframe(cmp, hide_index=True, width="stretch",
                         column_config={k: st.column_config.NumberColumn(format="%.1f") for k in ("MAE", "RMSE")}
                         | {"MAPE": st.column_config.NumberColumn("MAPE (%)", format="%.0f")})
            melhor = cmp.loc[cmp["MAPE"].idxmin(), "Modelo"]
            _md(leitura(
                f"Os dois modelos preveem <b>totais mensais</b> sobre o <b>mesmo período de teste</b>, então as métricas "
                f"são diretamente comparáveis. Menor MAPE: <b>{melhor}</b>.",
                "<span class='alerta'>Se os dois ficarem acima de 50%, o volume de desligamentos é dirigido por "
                "<b>eventos</b> (resultados de concursos, movimentações), não por um processo suave: trate a previsão "
                "como exercício metodológico, não como base de decisão.</span>"
                if cmp["MAPE"].min() > 50 else
                "Mesmo com erro aceitável, comunique sempre o <b>intervalo</b>, nunca só o número central."))

    # ---------------------------------------------------------------- 13. painel resumo
    with st.container(border=True):
        _md(secao("13 · Painel resumo", "Os quatro achados que sustentam uma decisão — a figura para levar à reunião"))
        dec = calc_decomposicao(idx, vals)
        an = calc_anomalias(idx, vals)
        f = make_subplots(rows=2, cols=2, vertical_spacing=0.16, horizontal_spacing=0.08,
                          subplot_titles=("Série mensal + tendência", "Anomalias",
                                          "Composição trimestral por categoria", "Prophet — próximos 12 meses"))
        f.add_scatter(x=s.index, y=s.values, mode="lines", line=dict(color=AZUL, width=1.3), fill="tozeroy",
                      fillcolor="rgba(42,120,214,.15)", name="Saídas/mês", row=1, col=1)
        if dec is not None:
            f.add_scatter(x=dec["tend"].index, y=dec["tend"], mode="lines", line=dict(color=MARINHO, width=2.8),
                          name="Tendência", row=1, col=1)
        a = an[an["anomalia"]]
        f.add_scatter(x=an.index, y=an["qtde"], mode="lines", line=dict(color=AZUL, width=1.1), showlegend=False, row=1, col=2)
        f.add_scatter(x=a.index, y=a["qtde"], mode="markers", name=f"Anomalias ({len(a)})",
                      marker=dict(symbol="triangle-down", size=11, color=CRITICO), row=1, col=2)
        dq = dfp[dfp["Periodo"] <= s.index[-1]]
        comp = (dq.groupby([dq["Periodo"].dt.to_period("Q").dt.start_time, "Categoria"]).size()
                .unstack(fill_value=0).reindex(columns=list(COR_CATEGORIA), fill_value=0))
        for c in comp.columns:
            f.add_bar(x=comp.index, y=comp[c], name=c, marker_color=COR_CATEGORIA[c],
                      marker_line=dict(color="#fff", width=1), row=2, col=1)
        ult = s.index[-1]
        hist = s[s.index > ult - pd.DateOffset(months=24)]
        fut = pr["fc_full"][pr["fc_full"]["ds"] > ult]
        f.add_scatter(x=fut["ds"], y=fut["yhat_upper"], mode="lines", line=dict(width=0), showlegend=False,
                      hoverinfo="skip", row=2, col=2)
        f.add_scatter(x=fut["ds"], y=fut["yhat_lower"], mode="lines", line=dict(width=0), fill="tonexty",
                      fillcolor="rgba(208,59,59,.15)", name="IC 95%", row=2, col=2)
        f.add_scatter(x=hist.index, y=hist.values, mode="markers", marker=dict(color=AZUL, size=6),
                      name="Últimos 24 meses", row=2, col=2)
        f.add_scatter(x=fut["ds"], y=fut["yhat"], mode="lines", line=dict(color=CRITICO, width=2.2),
                      name="Previsão 12 meses", row=2, col=2)
        tema(f, 640, barmode="stack", bargap=0.15, legend=dict(orientation="h", y=-0.08, yanchor="top", x=0))
        f.update_annotations(font=dict(size=12, color=INK_2))
        for r_ in (1, 2):
            for c_ in (1, 2):
                f.update_yaxes(gridcolor="#eef0f3", zeroline=False, tickfont=dict(color=MUTED), row=r_, col=c_)
        _g(f)
        _md(leitura(
            "Consolida em uma tela os quatro achados que sustentam qualquer decisão; os gráficos anteriores são o "
            "memorial de cálculo.",
            itens("<b>Série + tendência (↖)</b> — a série observada com a tendência da decomposição (gráfico 6). "
                  "Responde <i>a evasão está crescendo?</i> em uma olhada.",
                  f"<b>Anomalias (↗)</b> — os {len(a)} meses fora do padrão (gráfico 7). Responde <i>quantos eventos "
                  "excepcionais tivemos e quando?</i>",
                  "<b>Composição por categoria (↙)</b> — cada barra é um trimestre, dividido entre oficiais e praças. "
                  "Em barras empilhadas só o segmento de baixo tem leitura visual confiável; leia a <b>proporção</b> "
                  "dentro de cada trimestre, não a trajetória de cada categoria.",
                  "<b>Prophet — próximos 12 meses (↘)</b> — só a projeção verdadeira, feita pelo modelo "
                  "<b>reajustado com todos os meses</b> (gráfico 12), com os 24 meses mais recentes para referência. "
                  "Responde <i>o que esperar no próximo ano?</i>"),
            "<span class='alerta'><b>Ao apresentar:</b> a faixa de confiança é larga, e isso não é falha — é a incerteza "
            "honesta de projetar 12 meses. Com resíduos de caudas pesadas (gráfico 10), o cenário real é ainda mais "
            "amplo. Comunique o intervalo, nunca o número central sozinho — e lembre que a série é majoritariamente "
            "sintética.</span>"))


def _cartoes_metricas(m: dict) -> str:
    def c(rot, val, sub):
        return (f'<div class="kpi" style="padding:.6rem .8rem"><div class="kpi-label">{rot}</div>'
                f'<div class="kpi-value small">{val}</div><div class="kpi-ctx">{sub}</div></div>')
    return ('<div class="kpi-grid" style="grid-template-columns:repeat(3,minmax(0,1fr));margin:.4rem 0 .2rem">'
            + c("MAE", _num(m["MAE"]), "erro médio · saídas/mês")
            + c("RMSE", _num(m["RMSE"]), "penaliza erros grandes")
            + c("MAPE", f"{m['MAPE']:.0f}%", "erro percentual") + "</div>")
