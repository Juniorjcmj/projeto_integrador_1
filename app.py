"""Painel de evasão / desligamentos a pedido de militares.

Executar DE DENTRO da pasta dashboard/ (é onde o Streamlit procura .streamlit/secrets.toml):
    streamlit run app.py
"""
import importlib
import warnings
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import dados
importlib.reload(dados)
from dados import (
    MOTIVOS,
    NOMES_POSTOS,
    ORDEM_POSTOS,
    POSTOS_QUADRO,
    TEMAS,
    carregar,
    gravar_registro,
    proximo_id_pessoa,
    serie_mensal,
    url_painel_supabase,
)

import ia
importlib.reload(ia)
from ia import (
    DEFAULT_MODEL,
    analisar_entrevista_nemotron,
    extrair_texto_arquivo,
    get_system_prompt,
    ia_configurada,
    stream_nemotron,
)

import estilos
importlib.reload(estilos)

import analise_temporal
importlib.reload(analise_temporal)
from analise_temporal import aba_previsao, aba_series
from estilos import (
    AZUL,
    CONFIG_PLOTLY,
    COR_CATEGORIA,
    COR_MOTIVO,
    CSS_MODERNO,
    ESCALA_SEQ,
    INK,
    MARINHO,
    MUTED,
    achados,
    barra_origem,
    etapas,
    fonte_status,
    kpi,
    leitura,
    secao,
    tema,
)

st.set_page_config(page_title="Evasão de Militares · Painel", page_icon=":material/insights:", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(CSS_MODERNO, unsafe_allow_html=True)

MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
TEMAS_CURTOS = {
    "Concurso Público": "Concurso",
    "Saúde": "Saúde",
    "Família / Vida Pessoal": "Família",
    "Financeiro": "Financeiro",
    "Estudos / Pós-graduação": "Estudos",
    "Insatisfação Profissional": "Insatisfação",
}


@st.cache_data(ttl=300)
def _dados():
    df = carregar()
    return df, datetime.now()


def grafico(fig):
    st.plotly_chart(fig, width="stretch", config=CONFIG_PLOTLY)


def pct(x: float) -> str:
    return f"{x:.0%}".replace(".", ",")


def variacao(atual: float, anterior: float) -> tuple[str | None, str]:
    """Texto e tom do delta. Para evasão, subir é ruim."""
    if not anterior:
        return None, "neutro"
    v = atual / anterior - 1
    if abs(v) < 0.005:
        return "estável", "neutro"
    return (f"{'▲' if v > 0 else '▼'} {abs(v):.0%}", "ruim" if v > 0 else "bom")


df_all, carregado_em = _dados()
fonte = df_all.attrs.get("fonte", "")
erro_fonte = df_all.attrs.get("erro_fonte")
temas = list(TEMAS.values())
HOJE = pd.Timestamp.today().normalize().replace(day=1)
tem_ia = ia_configurada()

# ================================================================= sidebar
with st.sidebar:
    st.markdown(
        '<div class="sb-brand"><div class="sb-logo">EM</div><div>'
        '<div class="sb-name">Evasão de Militares</div><div class="sb-sub">Painel de retenção de talentos</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )

    status = []
    if fonte == "Supabase":
        status.append(("ok", "Banco Supabase conectado", f"{len(df_all)} registros carregados"))
    elif erro_fonte:
        status.append(("off", "Supabase indisponível", "Exibindo o backup local; novas entrevistas não serão gravadas"))
    else:
        status.append(("warn", "Modo local (Excel)", "Supabase não configurado; gravações vão para o Excel"))
    status.append(("ok", "Nemotron disponível", DEFAULT_MODEL.split("/")[-1]) if tem_ia
                  else ("off", "IA indisponível", "Defina NVIDIA_API_KEY no .env ou secrets"))
    st.markdown(fonte_status(status), unsafe_allow_html=True)

    link_banco = url_painel_supabase()
    c_a, c_b = st.columns(2) if link_banco else (st.container(), None)
    if c_a.button("Atualizar", icon=":material/refresh:", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    if link_banco:
        c_b.link_button("Banco", link_banco, icon=":material/database:", width="stretch")

    st.markdown('<div class="sb-label">Recorte da análise</div>', unsafe_allow_html=True)
    origem = st.segmented_control("Origem dos registros", ["Todos", "Reais", "Sintéticos"], default="Todos",
                                  help="Reais = entrevistas de saída. Sintéticos = gerados por modelo estatístico.")
    anos_disp = sorted(df_all["Periodo"].dropna().dt.year.unique().astype(int))
    anos_disp = [a for a in anos_disp if a <= HOJE.year] or anos_disp
    if len(anos_disp) > 1:
        ano_ini, ano_fim = st.select_slider("Período (anos)", options=anos_disp,
                                            value=(anos_disp[0], anos_disp[-1]))
    else:
        ano_ini = ano_fim = anos_disp[0]
    cats = st.pills("Categoria", ["Oficial", "Praça", "Não Informado"], selection_mode="multi",
                    default=["Oficial", "Praça", "Não Informado"])
    motivos_sel = st.multiselect("Motivo principal", MOTIVOS, default=MOTIVOS)

    with st.expander("Opções avançadas"):
        ocultar_futuro = st.toggle(f"Ocultar meses após {HOJE:%m/%Y}", value=True)
        janela = st.slider("Média móvel (meses)", 2, 12, 3)

# ================================================================= filtros
df = df_all.copy()
if origem == "Reais":
    df = df[df["Origem"] == "Real"]
elif origem == "Sintéticos":
    df = df[df["Origem"] == "Sintético"]
ano = df["Periodo"].dt.year
df = df[df["Periodo"].isna() | ano.between(ano_ini, ano_fim)]
if ocultar_futuro:
    df = df[df["Periodo"].isna() | (df["Periodo"] <= HOJE)]
df = df[df["Categoria"].isin(cats or []) & df["Motivo_Principal"].isin(motivos_sel)]
n_sem_data = int(df["Periodo"].isna().sum())
dfp = df.dropna(subset=["Periodo"])
n_real = int((df["Origem"] == "Real").sum())
n_sint = len(df) - n_real

# ================================================================= cabeçalho
if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"), icon=":material/check_circle:")

periodo_txt = f"{dfp['Periodo'].min():%m/%Y} – {dfp['Periodo'].max():%m/%Y}" if not dfp.empty else "—"
st.markdown(
    f"""
    <div class="hero">
      <div>
        <div class="eyebrow">Painel de evasão e retenção de militares</div>
        <h1>Por que os militares estão pedindo desligamento?</h1>
        <p>Desligamentos a pedido analisados a partir das entrevistas de saída: <b>quanto</b>, <b>por quê</b>,
        <b>quem</b> e <b>quando</b>. Novas entrevistas são lidas pela IA e entram no painel automaticamente.</p>
      </div>
      <div class="hero-meta">Período exibido: <b>{periodo_txt}</b><br>
      Dados atualizados às <b>{carregado_em:%H:%M}</b> de {carregado_em:%d/%m/%Y}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(barra_origem(n_real, n_sint), unsafe_allow_html=True)

if dfp.empty:
    st.info("Nenhum registro com os filtros atuais. Amplie o recorte na barra lateral.", icon=":material/filter_alt_off:")
    st.stop()

# ================================================================= métricas-chave
mensal = serie_mensal(dfp)
mensal["Média móvel"] = mensal["Qtde"].rolling(janela, min_periods=1).mean()
pico = mensal.loc[mensal["Qtde"].idxmax()]
total = len(dfp)

mot = dfp["Motivo_Principal"].value_counts(normalize=True)
postos = dfp.loc[dfp["Posto"] != "Não Informado", "Posto"].value_counts()
posto_top = postos.index[0] if len(postos) else "—"
posto_share = postos.iloc[0] / total if len(postos) else 0
temas_pct = dfp[temas].mean().sort_values(ascending=False)
cat_share = dfp["Categoria"].value_counts(normalize=True)

q = mensal["Qtde"].to_numpy()
ult12, ant12 = q[-12:].sum(), (q[-24:-12].sum() if len(q) >= 24 else 0)
ult6, ant6 = q[-6:].sum(), (q[-12:-6].sum() if len(q) >= 12 else 0)
d12, tom12 = variacao(ult12, ant12)

st.markdown(
    '<div class="kpi-grid">'
    + kpi("Desligamentos no recorte", f"{total:,}".replace(",", "."), f"entre {periodo_txt}")
    + kpi("Últimos 12 meses", f"{int(ult12)}", "vs. 12 meses anteriores" if d12 else "histórico curto para comparar",
          delta=d12, tom=tom12)
    + kpi("Motivo mais frequente", mot.index[0], f"{pct(mot.iloc[0])} das saídas", pequeno=True)
    + kpi("Posto mais afetado", f"{posto_top} · {NOMES_POSTOS.get(posto_top, posto_top)}",
          f"{pct(posto_share)} das saídas", pequeno=True)
    + "</div>",
    unsafe_allow_html=True,
)

# achados em linguagem natural
itens = []
seg = f"; em seguida vem <b>{mot.index[1]}</b> ({pct(mot.iloc[1])})" if len(mot) > 1 else ""
itens.append(f"<b>{mot.index[0]}</b> é o motivo principal em <b>{pct(mot.iloc[0])}</b> das saídas{seg}.")
if ant6:
    v6 = ult6 / ant6 - 1
    sentido = "acima" if v6 > 0 else "abaixo"
    itens.append(f"Nos últimos 6 meses foram <b>{int(ult6)}</b> desligamentos, <b>{abs(v6):.0%} {sentido}</b> "
                 f"dos 6 meses anteriores ({int(ant6)}).")
else:
    itens.append(f"O pico mensal foi em <b>{MESES[pico['Periodo'].month - 1]}/{pico['Periodo'].year}</b>, "
                 f"com <b>{int(pico['Qtde'])}</b> desligamentos.")
if len(postos) > 1:
    top2 = (postos.iloc[0] + postos.iloc[1]) / total
    itens.append(f"Oficiais são <b>{pct(cat_share.get('Oficial', 0))}</b> das saídas; <b>{postos.index[0]}</b> e "
                 f"<b>{postos.index[1]}</b> concentram <b>{pct(top2)}</b> do total.")
itens.append(f"<b>{temas_pct.index[0]}</b> é o tema mais citado nas entrevistas "
             f"(<b>{pct(temas_pct.iloc[0])}</b> dos entrevistados), seguido de {temas_pct.index[1]} "
             f"({pct(temas_pct.iloc[1])}).")
st.markdown(achados(itens), unsafe_allow_html=True)

if n_sem_data:
    st.caption(f"{n_sem_data} registro(s) sem data de desligamento ficam fora dos gráficos temporais.")

tab_geral, tab_por_que, tab_quem, tab_quando, tab_series, tab_prev, tab_nova, tab_ia = st.tabs([
    ":material/monitoring: Visão geral",
    ":material/help: Por que saem",
    ":material/groups: Quem sai",
    ":material/calendar_month: Quando saem",
    ":material/timeline: Séries temporais",
    ":material/online_prediction: Previsão",
    ":material/upload_file: Nova entrevista",
    ":material/smart_toy: Consultor IA",
])

# ================================================================= 1. visão geral
with tab_geral:
    c1, c2 = st.columns([1.65, 1], gap="medium")
    with c1, st.container(border=True):
        st.markdown(secao(
            f"Média de {mensal['Qtde'].mean():.1f} desligamentos por mês, com pico em "
            f"{MESES[pico['Periodo'].month - 1]}/{pico['Periodo'].year} ({int(pico['Qtde'])})",
            f"Barras: total mensal · Linha: média móvel de {janela} meses",
        ), unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_bar(x=mensal["Periodo"], y=mensal["Qtde"], name="Desligamentos no mês",
                    marker_color="#b7d3f6", hovertemplate="%{y} desligamentos<extra></extra>")
        fig.add_scatter(x=mensal["Periodo"], y=mensal["Média móvel"], name=f"Média móvel ({janela} meses)",
                        mode="lines", line=dict(color=MARINHO, width=2.5),
                        hovertemplate="média %{y:.1f}<extra></extra>")
        grafico(tema(fig, 360, hovermode="x unified", bargap=0.15, xaxis_hoverformat="%b/%Y"))
        st.markdown(leitura(
            "Cada <b>barra clara</b> é o número de militares que pediram desligamento naquele mês. A <b>linha escura</b> "
            f"é a média móvel de {janela} meses: ela suaviza os altos e baixos de um mês para o outro e mostra a "
            "direção geral da evasão (ajuste a janela em <i>Opções avançadas</i>).",
            f"<b>O que procurar:</b> se a linha sobe de forma sustentada, a evasão está aumentando — não é só um mês "
            f"ruim. O pico do período foi <b>{MESES[pico['Periodo'].month - 1]}/{pico['Periodo'].year}</b>, com "
            f"{int(pico['Qtde'])} saídas. Lembre que a maior parte dos registros é sintética: confirme a tendência "
            "no filtro <b>Reais</b>."), unsafe_allow_html=True)

    with c2, st.container(border=True):
        st.markdown(secao("O que motivou cada saída", "Motivo principal declarado · % do total"),
                    unsafe_allow_html=True)
        m = mot.sort_values().rename_axis("Motivo").reset_index(name="p")
        fig = go.Figure(go.Bar(
            x=m["p"], y=m["Motivo"], orientation="h",
            marker_color=[COR_MOTIVO.get(x, MUTED) for x in m["Motivo"]],
            text=[pct(v) for v in m["p"]], textposition="outside", cliponaxis=False,
            textfont=dict(color=INK, size=12), hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        ))
        grafico(tema(fig, 360, xaxis=dict(visible=False, range=[0, m["p"].max() * 1.22]),
                     yaxis=dict(tickfont=dict(color=INK, size=12)), bargap=0.5))
        st.markdown(leitura(
            "Cada barra é a fatia das saídas atribuída a um <b>motivo principal</b> — o motivo que o militar apontou "
            "como decisivo. As fatias somam 100%.",
            f"<b>{mot.index[0]}</b> domina com {pct(mot.iloc[0])}. Quanto mais concentrado em um motivo, mais "
            "focada pode ser a ação de retenção. “Não Informado” alto indica entrevistas pouco conclusivas."),
            unsafe_allow_html=True)

    with st.container(border=True):
        anual = dfp.groupby(dfp["Periodo"].dt.year).size().rename_axis("Ano").reset_index(name="Qtde")
        ano_atual_incompleto = anual["Ano"].max() == HOJE.year
        st.markdown(secao(
            "Volume anual de desligamentos",
            f"{HOJE.year} ainda está em andamento — o total do ano é parcial." if ano_atual_incompleto else "Total por ano",
        ), unsafe_allow_html=True)
        cores = [("#b7d3f6" if (a == HOJE.year) else AZUL) for a in anual["Ano"]]
        fig = go.Figure(go.Bar(x=anual["Ano"].astype(str), y=anual["Qtde"], marker_color=cores,
                               text=anual["Qtde"], textposition="outside", cliponaxis=False,
                               textfont=dict(color=INK), hovertemplate="%{x}: %{y} desligamentos<extra></extra>"))
        grafico(tema(fig, 260, yaxis=dict(visible=False, range=[0, anual["Qtde"].max() * 1.2]), bargap=0.45))
        st.markdown(leitura(
            "Total de desligamentos em cada ano. É a leitura mais simples de <b>escala</b>: o problema está "
            "crescendo de um ano para o outro?",
            "A barra mais clara é o ano corrente, ainda <b>incompleto</b> — ela não deve ser comparada diretamente "
            "com anos fechados." if ano_atual_incompleto else
            "Todos os anos exibidos estão completos e podem ser comparados diretamente."), unsafe_allow_html=True)

# ================================================================= 2. por que saem
with tab_por_que:
    with st.container(border=True):
        h1, h2 = st.columns([3, 1.3])
        with h2:
            gran = st.segmented_control("Agrupar por", ["Ano", "Semestre", "Trimestre"], default="Ano",
                                        label_visibility="collapsed") or "Ano"
        with h1:
            st.markdown(secao("Como a composição dos motivos mudou ao longo do tempo",
                              "Participação de cada motivo principal nas saídas do período (%)"),
                        unsafe_allow_html=True)
        mm = serie_mensal(dfp, "Motivo_Principal")
        p = mm["Periodo"]
        if gran == "Trimestre":
            mm["Período"] = p.dt.year.astype(str) + " T" + p.dt.quarter.astype(str)
        elif gran == "Semestre":
            mm["Período"] = p.dt.year.astype(str) + " S" + ((p.dt.month > 6) + 1).astype(str)
        else:
            mm["Período"] = p.dt.year.astype(str)
        mm = mm.groupby(["Período", "Motivo_Principal"], as_index=False)["Qtde"].sum()
        mm["%"] = mm["Qtde"] / mm.groupby("Período")["Qtde"].transform("sum")
        fig = px.bar(mm, x="Período", y="%", color="Motivo_Principal", color_discrete_map=COR_MOTIVO,
                     category_orders={"Motivo_Principal": MOTIVOS}, custom_data=["Qtde"])
        fig.update_traces(marker_line=dict(color="#ffffff", width=2),
                          hovertemplate="%{fullData.name}: %{y:.0%} (%{customdata[0]})<extra></extra>")
        grafico(tema(fig, 360, yaxis=dict(tickformat=".0%", gridcolor="#eef0f3"), barcornerradius=0,
                     hovermode="x unified", bargap=0.3))
        st.markdown(leitura(
            "Cada barra é um período e sempre soma 100%; as cores mostram quanto cada motivo principal representou "
            "nas saídas daquele período. O gráfico mostra <b>proporção</b>, não quantidade — para volumes, veja a "
            "Visão geral.",
            "<b>O que procurar:</b> uma cor que cresce de um período para o outro indica um motivo ganhando peso. "
            "Por exemplo, se “Família / Vida Pessoal” aumenta, pesam cada vez mais as questões de movimentação e "
            "vida familiar. Use o seletor para ver por ano, semestre ou trimestre (períodos menores têm menos casos "
            "e variam mais)."), unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(border=True):
        st.markdown(secao(
            f"{temas_pct.index[0]} aparece em {pct(temas_pct.iloc[0])} das entrevistas",
            "% dos entrevistados que mencionam cada tema — uma pessoa pode citar vários",
        ), unsafe_allow_html=True)
        t = temas_pct.sort_values()
        fig = go.Figure(go.Bar(x=t.values, y=t.index, orientation="h", marker_color=AZUL,
                               text=[pct(v) for v in t.values], textposition="outside", cliponaxis=False,
                               textfont=dict(color=INK), hovertemplate="%{y}: %{x:.0%}<extra></extra>"))
        grafico(tema(fig, 330, xaxis=dict(visible=False, range=[0, max(t.max(), 0.01) * 1.22]),
                     yaxis=dict(tickfont=dict(color=INK)), bargap=0.4))
        st.markdown(leitura(
            "Percentual de entrevistados que <b>mencionaram</b> cada tema, mesmo que não seja o motivo principal. "
            "Uma pessoa pode citar vários temas, por isso as barras <b>não somam 100%</b>.",
            "É a leitura <b>multicausal</b> da evasão: um tema muito citado mas pouco escolhido como motivo principal "
            "funciona como fator de fundo — sozinho não faz o militar sair, mas pesa na decisão."),
            unsafe_allow_html=True)

    with c2, st.container(border=True):
        st.markdown(secao("Menção a cada tema, ano a ano", "% dos entrevistados do ano que citam o tema"),
                    unsafe_allow_html=True)
        heat = dfp.groupby(dfp["Periodo"].dt.year)[temas].mean().mul(100).T
        heat.index = [TEMAS_CURTOS[i] for i in heat.index]
        heat.columns = heat.columns.astype(str)
        fig = px.imshow(heat, text_auto=".0f", aspect="auto", color_continuous_scale=ESCALA_SEQ, zmin=0, zmax=100)
        fig.update_traces(hovertemplate="%{y} em %{x}: %{z:.0f}%<extra></extra>", xgap=2, ygap=2)
        grafico(tema(fig, 330, coloraxis_showscale=False, xaxis=dict(side="top"),
                     yaxis=dict(tickfont=dict(color=INK))))
        st.markdown(leitura(
            "Cada célula é o % de entrevistados daquele ano que citou o tema; cor mais escura = mais citado. "
            "Leia <b>cada linha da esquerda para a direita</b> para ver se um tema está ganhando ou perdendo força.",
            "Anos com poucas entrevistas variam mais — uma mudança grande de um ano para o outro pode ser efeito de "
            "amostra pequena, não uma mudança real."), unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(secao("Quais temas aparecem juntos na mesma entrevista",
                          "% dos entrevistados que citam os dois temas · a diagonal é o total de cada tema"),
                    unsafe_allow_html=True)
        b = dfp[temas].astype(int)
        co = b.T.dot(b) / len(b) * 100
        co.index = co.columns = [TEMAS_CURTOS[i] for i in co.index]
        fig = px.imshow(co, text_auto=".0f", aspect="auto", color_continuous_scale=ESCALA_SEQ, zmin=0)
        fig.update_traces(hovertemplate="%{y} + %{x}: %{z:.0f}%<extra></extra>", xgap=2, ygap=2)
        grafico(tema(fig, 360, coloraxis_showscale=False, xaxis=dict(side="top"),
                     yaxis=dict(tickfont=dict(color=INK))))
        st.markdown(leitura(
            "Cada célula fora da diagonal mostra o % de entrevistados que citou <b>os dois temas ao mesmo tempo</b>; "
            "a diagonal é o total de cada tema sozinho. A matriz é simétrica (Concurso + Família = Família + Concurso).",
            "<b>O que procurar:</b> pares com valor alto revelam <b>combinações de causas</b>. Por exemplo, "
            "Concurso + Estudos sugere uma saída planejada; Família + Insatisfação sugere desgaste com a carreira. "
            "Combinações frequentes pedem ações que atuem nas duas causas juntas."), unsafe_allow_html=True)

# ================================================================= 3. quem sai
with tab_quem:
    c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(border=True):
        st.markdown(secao(
            f"{NOMES_POSTOS.get(posto_top, posto_top)} ({posto_top}) é o posto com mais desligamentos",
            "Quantidade por posto/graduação, na ordem hierárquica",
        ), unsafe_allow_html=True)
        pp = dfp["Posto"].value_counts().reindex(ORDEM_POSTOS).dropna()
        fig = go.Figure(go.Bar(
            x=pp.index, y=pp.values, text=pp.values.astype(int), textposition="outside", cliponaxis=False,
            marker_color=[MARINHO if x == posto_top else "#86b6ef" for x in pp.index], textfont=dict(color=INK),
            customdata=[NOMES_POSTOS.get(x, x) for x in pp.index],
            hovertemplate="%{customdata}: %{y}<extra></extra>",
        ))
        grafico(tema(fig, 330, yaxis=dict(visible=False, range=[0, pp.max() * 1.18]), bargap=0.35))
        st.markdown(leitura(
            "Número de desligamentos em cada posto/graduação, na <b>ordem hierárquica</b> (do mais moderno ao mais "
            f"antigo). O posto com mais saídas está destacado: <b>{posto_top}</b>.",
            "É uma contagem absoluta: postos com mais militares tendem a ter mais saídas. Para comparar a "
            "<b>intensidade</b> entre postos, veja o gráfico 2 da aba <b>Séries temporais</b>, que usa percentuais."),
            unsafe_allow_html=True)

    with c2, st.container(border=True):
        st.markdown(secao(
            f"Oficiais respondem por {pct(cat_share.get('Oficial', 0))} das saídas",
            "Desligamentos por ano e categoria",
        ), unsafe_allow_html=True)
        cc = dfp.groupby([dfp["Periodo"].dt.year.rename("Ano"), "Categoria"]).size().reset_index(name="Qtde")
        cc["Ano"] = cc["Ano"].astype(str)
        fig = px.bar(cc, x="Ano", y="Qtde", color="Categoria", color_discrete_map=COR_CATEGORIA,
                     category_orders={"Categoria": list(COR_CATEGORIA)})
        fig.update_traces(marker_line=dict(color="#ffffff", width=2),
                          hovertemplate="%{fullData.name}: %{y}<extra></extra>")
        grafico(tema(fig, 330, barcornerradius=0, hovermode="x unified", bargap=0.4))
        st.markdown(leitura(
            "Cada barra é um ano, dividida entre <b>oficiais</b> e <b>praças</b>. A altura total é o volume do ano; "
            "os segmentos mostram quanto veio de cada categoria.",
            "Em barras empilhadas, só o segmento de baixo tem comparação visual confiável entre anos — o de cima "
            "“flutua”. Leia a <b>proporção</b> dentro de cada ano e use o valor exato no hover."),
            unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(secao("Motivo principal dentro de cada posto",
                          "% das saídas de cada posto (cada linha soma 100%) · n = total de saídas do posto"),
                    unsafe_allow_html=True)
        mp = pd.crosstab(dfp["Posto"], dfp["Motivo_Principal"]).reindex(ORDEM_POSTOS).dropna(how="all")
        mp = mp[[c for c in MOTIVOS if c in mp.columns]]
        n = mp.sum(axis=1)
        mp_pct = mp.div(n, axis=0).mul(100)
        mp_pct.index = [f"{i}  (n={int(k)})" for i, k in zip(mp_pct.index, n)]
        fig = px.imshow(mp_pct, text_auto=".0f", aspect="auto", color_continuous_scale=ESCALA_SEQ, zmin=0, zmax=100)
        fig.update_traces(hovertemplate="%{y} · %{x}: %{z:.0f}%<extra></extra>", xgap=2, ygap=2)
        grafico(tema(fig, 380, coloraxis_showscale=False, xaxis=dict(side="top", tickfont=dict(color=INK)),
                     yaxis=dict(tickfont=dict(color=INK))))
        st.markdown(leitura(
            "Cada linha é um posto e <b>soma 100%</b>: as células mostram como as saídas daquele posto se dividem "
            "entre os motivos. Por ser normalizado por linha, postos pequenos e grandes ficam comparáveis; o "
            "<b>n</b> ao lado indica quantos casos sustentam cada linha.",
            "<b>O que procurar:</b> um motivo que se destaca em um posto e não nos outros — sinal de uma causa "
            "ligada àquela fase da carreira (ex.: concurso público pesando mais nos postos iniciais). Desconfie de "
            "linhas com n pequeno: um ou dois casos mudam muito o percentual."), unsafe_allow_html=True)

# ================================================================= 4. quando saem
with tab_quando:
    with st.container(border=True):
        s = mensal.assign(Ano=mensal["Periodo"].dt.year, Mês=mensal["Periodo"].dt.month)
        media_mes = s.groupby("Mês")["Qtde"].mean()
        mes_top = MESES[int(media_mes.idxmax()) - 1]
        st.markdown(secao(
            f"{mes_top} é o mês com mais desligamentos: média de {media_mes.max():.0f} por ano",
            "Desligamentos por mês e ano — tons mais escuros indicam mais saídas",
        ), unsafe_allow_html=True)
        piv = s.pivot(index="Ano", columns="Mês", values="Qtde").reindex(columns=range(1, 13))
        piv.columns = MESES
        piv.index = piv.index.astype(str)
        fig = px.imshow(piv, text_auto=True, aspect="auto", color_continuous_scale=ESCALA_SEQ)
        fig.update_traces(hovertemplate="%{x}/%{y}: %{z} desligamentos<extra></extra>", xgap=2, ygap=2)
        grafico(tema(fig, 300, coloraxis_showscale=False, xaxis=dict(side="top", tickfont=dict(color=INK)),
                     yaxis=dict(tickfont=dict(color=INK))))
        st.markdown(leitura(
            "Cada linha é um ano e cada coluna um mês; o número é a quantidade de saídas e a cor, a intensidade "
            "(mais escuro = mais saídas).",
            "<b>O que procurar:</b> uma <b>coluna escura em vários anos</b> indica um mês que se repete como pico — um "
            "padrão de calendário (ex.: datas de posse em concursos). Uma célula escura isolada é um evento "
            "pontual. Para separar os dois com rigor estatístico, veja os gráficos 5 e 6 da aba "
            "<b>Séries temporais</b>."), unsafe_allow_html=True)

    with st.container(border=True):
        h1, h2 = st.columns([3, 1.2])
        with h2:
            horizonte = st.slider("Meses à frente", 3, 12, 6)
        if len(mensal) < 18:
            with h1:
                st.markdown(secao("Projeção indisponível",
                                  "São necessários pelo menos 18 meses de histórico no recorte."),
                            unsafe_allow_html=True)
        else:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            y = mensal.set_index("Periodo")["Qtde"].astype(float)
            y.index.freq = "MS"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ajuste = ExponentialSmoothing(y, trend="add", damped_trend=True).fit()
            prev = ajuste.forecast(horizonte).clip(lower=0)
            resid = (y - ajuste.fittedvalues).std()
            with h1:
                st.markdown(secao(
                    f"Projeção de referência: cerca de {prev.mean():.0f} desligamentos por mês "
                    f"nos próximos {horizonte} meses",
                    "Modelo de Holt (tendência amortecida) · faixa sombreada ≈ intervalo de 95%",
                ), unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_scatter(x=y.index, y=y, name="Observado", mode="lines", line=dict(color=AZUL, width=2),
                            hovertemplate="%{y:.0f}<extra>Observado</extra>")
            fig.add_scatter(x=prev.index, y=prev + 1.96 * resid, mode="lines", line=dict(width=0),
                            showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=prev.index, y=(prev - 1.96 * resid).clip(lower=0), mode="lines", fill="tonexty",
                            fillcolor="rgba(11, 59, 117, 0.10)", line=dict(width=0), name="Faixa ~95%",
                            hoverinfo="skip")
            prev = pd.concat([y.iloc[-1:], prev])  # liga a projeção ao último mês observado
            fig.add_scatter(x=prev.index, y=prev, name="Projeção", mode="lines",
                            line=dict(dash="dash", color=MARINHO, width=2.5),
                            hovertemplate="%{y:.1f}<extra>Projeção</extra>")
            grafico(tema(fig, 340, hovermode="x unified", xaxis_hoverformat="%b/%Y"))
            st.markdown(leitura(
                "A <b>linha azul</b> é o histórico mensal; a <b>tracejada</b>, a projeção do modelo de Holt, que segue a "
                "tendência recente e a amortece com o tempo (para não extrapolar uma alta ou queda indefinidamente). "
                "A <b>faixa sombreada</b> é o intervalo onde o valor real deve cair em ~95% dos casos.",
                "Use como <b>referência de ordem de grandeza</b>, não como previsão: a série é curta e majoritariamente "
                "sintética. Para modelos completos, com validação e comparação (ARIMA e Prophet), veja a aba "
                "<b>Previsão</b>."), unsafe_allow_html=True)

# ================================================================= 5–6. séries temporais e previsão
with tab_series:
    aba_series(df_all, df, dfp, HOJE, janela, ocultar_futuro)

with tab_prev:
    aba_previsao(dfp, HOJE)

# ================================================================= 7. nova entrevista
with tab_nova:
    passo = st.empty()
    prox_id = proximo_id_pessoa(df_all)
    extraido = st.session_state.get("novo_registro_extraido")

    texto_entrevista = ""
    if not extraido:
        c_in, c_side = st.columns([1.7, 1], gap="medium")
        with c_in, st.container(border=True):
            st.markdown(secao("Envie a entrevista de desligamento",
                              "Um arquivo ou o texto colado — o que tiver conteúdo será usado"),
                        unsafe_allow_html=True)
            modo = st.segmented_control("Entrada", ["Arquivo", "Colar texto"], default="Arquivo",
                                        label_visibility="collapsed") or "Arquivo"
            if modo == "Arquivo":
                arq = st.file_uploader("Arquivo da entrevista", type=["pdf", "docx", "txt"],
                                       label_visibility="collapsed")
                if arq is not None:
                    try:
                        texto_entrevista = extrair_texto_arquivo(arq).strip()
                        if texto_entrevista:
                            st.caption(f"✓ {arq.name} · {len(texto_entrevista):,} caracteres lidos".replace(",", "."))
                            with st.expander("Conferir texto extraído"):
                                st.text(texto_entrevista[:3000] + ("…" if len(texto_entrevista) > 3000 else ""))
                        else:
                            st.warning("Não foi possível extrair texto do arquivo. Se for um PDF escaneado, "
                                       "cole o texto manualmente.")
                    except Exception as e:
                        st.error(f"Erro ao ler o arquivo: {e}")
            else:
                texto_entrevista = st.text_area(
                    "Texto da entrevista", height=220, label_visibility="collapsed",
                    placeholder="Relatório de Desligamento a Pedido…\nMilitar: Primeiro-Tenente…\nData: …\nDeclarações: …",
                ).strip()

            clicou = st.button("Extrair campos com Nemotron", type="primary", icon=":material/auto_awesome:",
                               width="stretch", disabled=not (texto_entrevista and tem_ia))
            if not tem_ia:
                st.caption("IA indisponível: configure a NVIDIA_API_KEY para habilitar a extração.")

        with c_side, st.container(border=True):
            st.markdown(secao("Como funciona"), unsafe_allow_html=True)
            st.markdown(
                f"""
1. O Nemotron lê a entrevista e preenche as **12 colunas** da planilha: data, posto, motivo principal e os 6 temas citados.
2. Você **revisa e corrige** o que for preciso — nada é gravado sem sua confirmação.
3. Ao gravar, o registro entra na planilha e **todos os gráficos se atualizam**.

O novo registro receberá o ID **`{prox_id}`**.
                """
            )

        passo.markdown(etapas(2 if texto_entrevista else 1), unsafe_allow_html=True)
        if clicou:
            with st.status("Nemotron lendo a entrevista e classificando…", expanded=False) as sb:
                try:
                    reg, thinking, _ = analisar_entrevista_nemotron(texto_entrevista, proximo_id=prox_id)
                    st.session_state["novo_registro_extraido"] = reg
                    st.session_state["thinking_extracao"] = thinking
                    sb.update(label="Campos extraídos", state="complete")
                    st.rerun()
                except Exception as e:
                    sb.update(label="Falha na extração", state="error")
                    st.error(f"Não foi possível processar com a API da NVIDIA: {e}")

    if extraido:
        passo.markdown(etapas(3), unsafe_allow_html=True)
        d = extraido
        with st.container(border=True):
            st.markdown(secao(f"Revise o registro {d['ID_Pessoa']} antes de gravar",
                              "Campos pré-preenchidos pelo Nemotron — ajuste o que estiver incorreto"),
                        unsafe_allow_html=True)
            if d.get("Sintese_Entrevista"):
                just = (f"<small><b>Por que esta classificação:</b> {d['Justificativa_Classificacao']}</small>"
                        if d.get("Justificativa_Classificacao") else "")
                st.markdown(f'<div class="sintese">{d["Sintese_Entrevista"]}{just}</div>', unsafe_allow_html=True)
            if d.get("_faltantes"):
                nomes = {"Data_Desligamento": "data do desligamento", "Motivo_Principal": "motivo principal",
                         "Posto_Quadro": "posto/quadro"}
                faltam = ", ".join(nomes.get(f, f) for f in d["_faltantes"])
                st.markdown(f'<div class="aviso-campos"><b>Atenção:</b> a entrevista não deixa claro: {faltam}. '
                            "Um valor padrão foi sugerido — confira abaixo.</div>", unsafe_allow_html=True)

            with st.form("form_gravar_registro", border=False):
                f1, f2, f3, f4 = st.columns([1, 1, 1.2, 1.4])
                f1.text_input("ID", value=d["ID_Pessoa"], disabled=True)
                val_data = f2.date_input("Data do desligamento", value=d["Data_Desligamento"], format="DD/MM/YYYY")
                opcoes_posto = POSTOS_QUADRO if d["Posto_Quadro"] in POSTOS_QUADRO else [d["Posto_Quadro"], *POSTOS_QUADRO]
                val_posto = f3.selectbox("Posto / Quadro", opcoes_posto, index=opcoes_posto.index(d["Posto_Quadro"]))
                val_motivo = f4.selectbox("Motivo principal", MOTIVOS, index=MOTIVOS.index(d["Motivo_Principal"]))

                st.markdown('<div class="sec-s" style="margin:.4rem 0 .3rem">Temas citados na entrevista</div>',
                            unsafe_allow_html=True)
                cols = st.columns(3)
                marcados = {}
                for i, (campo, nome) in enumerate(TEMAS.items()):
                    marcados[campo] = cols[i % 3].checkbox(nome, value=bool(d.get(campo)))

                b1, b2 = st.columns([3, 1])
                salvar = b1.form_submit_button("Gravar na planilha", type="primary", icon=":material/save:",
                                               width="stretch")
                descartar = b2.form_submit_button("Descartar", width="stretch")

            if descartar:
                st.session_state.pop("novo_registro_extraido", None)
                st.rerun()
            if salvar:
                registro_final = {
                    "ID_Pessoa": d["ID_Pessoa"],
                    "Ano": val_data.year,
                    "Mes": val_data.month,
                    "Data_Desligamento": val_data.isoformat(),
                    "Posto_Quadro": val_posto,
                    "Motivo_Principal": val_motivo,
                    **marcados,
                }
                res = gravar_registro(registro_final)
                if res["ok"]:
                    destino = ("no banco Supabase (com cópia no Excel local)" if res["destino"] == "supabase"
                               else "no Excel local")
                    aviso = f" Atenção: {res['aviso']}" if res["aviso"] else ""
                    st.session_state["flash"] = (f"Registro {d['ID_Pessoa']} gravado {destino}. "
                                                 f"Os gráficos já incluem a nova entrevista.{aviso}")
                    st.session_state.pop("novo_registro_extraido", None)
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(f"Não foi gravado. {res['mensagem']}")

            if st.session_state.get("thinking_extracao"):
                with st.expander("Ver raciocínio do modelo"):
                    st.markdown(st.session_state["thinking_extracao"])

    if fonte != "Supabase":
        with st.expander("Como conectar ao banco Supabase"):
            st.markdown("""
Sem o banco, cada registro é gravado em `dataset_sintetico.xlsx`. Para usar o Supabase:

1. No SQL Editor do projeto Supabase, rode `supabase/schema.sql`.
2. Em `.streamlit/secrets.toml`, defina `SUPABASE_URL` e `SUPABASE_SECRET_KEY` (chave secreta, nunca a publicável).
3. Rode `python supabase/importar_excel.py` para carregar os registros existentes e clique em **Atualizar**.
            """)

# ================================================================= 6. consultor ia
with tab_ia:
    if not tem_ia:
        st.info("Configure a NVIDIA_API_KEY no `.env` ou em `.streamlit/secrets.toml` para usar o consultor.",
                icon=":material/key:")
    top1, top2 = st.columns([3, 1])
    with top1:
        st.markdown(secao("Pergunte aos dados em linguagem natural",
                          f"O Nemotron recebe um resumo do recorte atual ({total} registros, {periodo_txt}) "
                          "e responde como consultor de People Analytics"), unsafe_allow_html=True)
    mostrar_thinking = top2.toggle("Mostrar raciocínio", value=False)

    def responder(mensagens, alvo):
        caixa = alvo.status("Raciocinando…", expanded=True) if mostrar_thinking else None
        saida = alvo.empty()
        pensou = texto = ""
        for tipo, chunk in stream_nemotron(mensagens, enable_thinking=mostrar_thinking):
            if tipo == "reasoning" and caixa:
                pensou += chunk
                caixa.markdown(pensou)
            elif tipo == "content":
                texto += chunk
                saida.markdown(texto)
        if caixa:
            caixa.update(label="Raciocínio", state="complete", expanded=False)
        return texto

    with st.container(border=True):
        st.markdown(secao("Parecer executivo",
                          "Diagnóstico, grupos críticos, cautelas metodológicas e 3 a 5 recomendações"),
                    unsafe_allow_html=True)
        if st.button("Gerar parecer do recorte atual", type="primary", icon=":material/description:",
                     disabled=not tem_ia):
            prompt = (
                "Com base no cenário de dados filtrados, elabore um parecer executivo com:\n"
                "1. **Diagnóstico**: visão geral dos desligamentos e pontos de atenção imediatos.\n"
                "2. **Grupos mais críticos**: postos mais impactados e seus motivos principais.\n"
                "3. **Cautela metodológica**: o que muda ao separar dados reais de sintéticos.\n"
                "4. **Recomendações**: 3 a 5 medidas práticas para aumentar a retenção."
            )
            try:
                st.session_state["parecer"] = responder(
                    [{"role": "system", "content": get_system_prompt(df)}, {"role": "user", "content": prompt}], st)
            except Exception as e:
                st.error(f"Erro ao consultar a API da NVIDIA: {e}")
        elif st.session_state.get("parecer"):
            st.markdown(st.session_state["parecer"])

    with st.container(border=True):
        st.markdown(secao("Conversa"), unsafe_allow_html=True)
        hist = st.session_state.setdefault("chat_historico", [])
        sugestoes = ["Por que CTs saem mais?",
                     "E só com dados reais?",
                     "Sinais de insatisfação?"]
        pergunta = None
        if not hist:
            cs = st.columns(3)
            for c, sug in zip(cs, sugestoes):
                if c.button(sug, width="stretch", disabled=not tem_ia):
                    pergunta = sug
        for msg in hist:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        digitada = st.chat_input("Pergunte algo sobre os dados de evasão…", disabled=not tem_ia)
        pergunta = digitada or pergunta
        if pergunta:
            hist.append({"role": "user", "content": pergunta})
            with st.chat_message("user"):
                st.markdown(pergunta)
            msgs = [{"role": "system", "content": get_system_prompt(df)}] + hist[-6:]
            with st.chat_message("assistant"):
                try:
                    hist.append({"role": "assistant", "content": responder(msgs, st)})
                except Exception as e:
                    st.error(f"Erro ao consultar a API da NVIDIA: {e}")
        if hist and st.button("Limpar conversa", icon=":material/delete:"):
            st.session_state.chat_historico = []
            st.rerun()

# ================================================================= tabela
with st.expander(f"Ver os {len(df)} registros do recorte em tabela"):
    st.dataframe(df.drop(columns=["Data_Desligamento"]), width="stretch", hide_index=True)
    st.download_button("Baixar CSV", df.to_csv(index=False).encode("utf-8-sig"), "desligamentos_filtrado.csv",
                       icon=":material/download:")
