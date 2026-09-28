"""Design system do painel de evasão de militares.

Princípios: uma cor de marca (azul-marinho) para interface, cores de dados fixas por
entidade (um motivo tem sempre a mesma cor), tipografia única (Inter), grade e eixos
recessivos, e títulos que dizem a conclusão do gráfico — não apenas o nome dele.
"""
from html import escape

import plotly.graph_objects as go

# ------------------------------------------------------------------ tokens
INK = "#0b1220"
INK_2 = "#475569"
MUTED = "#8a94a6"
BORDER = "#e6e8ec"
GRID = "#eef0f3"
SURFACE = "#ffffff"
PAGE = "#f5f6f8"
MARINHO = "#0b3b75"
AZUL = "#2a78d6"
NEUTRO = "#b4b2a9"

# Cores de dados (paleta categórica validada p/ daltonismo, ordem fixa).
COR_MOTIVO = {
    "Concurso Público": "#2a78d6",
    "Família / Vida Pessoal": "#eb6834",
    "Saúde": "#1baf7a",
    "Financeiro": "#eda100",
    "Outro": "#4a3aa7",
    "Não Informado": NEUTRO,
}
COR_CATEGORIA = {"Oficial": "#2a78d6", "Praça": "#eb6834", "Não Informado": NEUTRO}
ESCALA_SEQ = ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

CSS_MODERNO = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, .stApp, .stMarkdown, button, input, textarea, select, label, p, li {{
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
[data-testid="stIconMaterial"], .material-symbols-rounded {{ font-family: 'Material Symbols Rounded' !important; }}
.stApp {{ background: {PAGE}; color: {INK}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
footer, #MainMenu {{ visibility: hidden; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1320px; }}

h1, h2, h3, h4 {{ color: {INK} !important; letter-spacing: -0.02em; font-weight: 700 !important; }}
p, li {{ color: {INK_2}; }}

/* ---------- sidebar */
section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.5rem; }}
.sb-brand {{ display:flex; gap:10px; align-items:center; margin-bottom: 1.1rem; }}
.sb-logo {{ width:34px; height:34px; border-radius:9px; background:{MARINHO}; color:#fff;
           display:flex; align-items:center; justify-content:center; font-weight:700; font-size:.95rem; }}
.sb-name {{ font-weight:700; color:{INK}; font-size:.95rem; line-height:1.15; }}
.sb-sub {{ color:{MUTED}; font-size:.75rem; }}
.sb-label {{ font-size:.7rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase;
            color:{MUTED}; margin: 1.1rem 0 .35rem; }}
.fonte-card {{ border:1px solid {BORDER}; border-radius:10px; padding:.7rem .8rem; background:#fafbfc; }}
.fonte-linha {{ display:flex; align-items:center; gap:8px; font-size:.82rem; color:{INK}; font-weight:500; }}
.fonte-linha + .fonte-linha {{ margin-top:.4rem; }}
.fonte-det {{ color:{MUTED}; font-size:.74rem; margin: .1rem 0 0 16px; }}
.dot {{ width:8px; height:8px; border-radius:50%; flex:none; }}
.dot.ok {{ background:#0ca30c; }} .dot.warn {{ background:#fab219; }} .dot.off {{ background:#d03b3b; }}

/* ---------- cabeçalho */
.hero {{ display:flex; justify-content:space-between; align-items:flex-end; gap:1.5rem; flex-wrap:wrap;
        margin-bottom: 1.4rem; }}
.eyebrow {{ font-size:.72rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:{MARINHO}; }}
.hero h1 {{ font-size: 2rem !important; margin: .3rem 0 .35rem !important; padding:0 !important; line-height:1.15; }}
.hero > div:first-child {{ flex: 1 1 560px; }}
.hero p {{ margin:0; font-size:.98rem; max-width: 720px; }}
.hero-meta {{ color:{MUTED}; font-size:.8rem; text-align:right; line-height:1.5; }}
.hero-meta b {{ color:{INK}; font-weight:600; }}

/* ---------- barra real x sintético */
.origem {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px; padding: .85rem 1rem;
          margin-bottom: 1.2rem; display:flex; gap:1.2rem; align-items:center; flex-wrap:wrap; }}
.origem-txt {{ font-size:.86rem; color:{INK_2}; flex: 1 1 380px; }}
.origem-txt b {{ color:{INK}; }}
.origem-bar {{ flex: 0 1 300px; min-width: 200px; }}
.origem-track {{ display:flex; height:8px; border-radius:4px; overflow:hidden; gap:2px; }}
.origem-leg {{ display:flex; justify-content:space-between; font-size:.74rem; color:{MUTED}; margin-top:.35rem; }}

/* ---------- KPIs */
.kpi-grid {{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: .9rem; margin-bottom: 1.2rem; }}
@media (max-width: 1100px) {{ .kpi-grid {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}
@media (max-width: 600px)  {{ .kpi-grid {{ grid-template-columns: 1fr; }} }}
.kpi {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px; padding: 1rem 1.1rem; }}
.kpi-label {{ font-size:.8rem; color:{INK_2}; font-weight:500; }}
.kpi-value {{ font-size:1.9rem; font-weight:700; color:{INK}; letter-spacing:-.02em; margin:.3rem 0 .2rem; line-height:1.15;
             overflow-wrap:anywhere; }}
.kpi-value.small {{ font-size:1.35rem; padding: .2rem 0 .15rem; }}
.kpi-ctx {{ font-size:.8rem; color:{MUTED}; }}
.delta {{ font-weight:600; font-size:.8rem; padding: 1px 6px; border-radius:5px; margin-right:4px; }}
.delta.ruim {{ color:#b42318; background:#fdecea; }}
.delta.bom {{ color:#006300; background:#e8f5e8; }}
.delta.neutro {{ color:{INK_2}; background:#f1f3f5; }}

/* ---------- achados */
.achados {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px; padding:1rem 1.2rem 1.1rem;
           margin-bottom: 1.6rem; }}
.achados-tit {{ font-size:.72rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:{MARINHO};
               margin-bottom:.6rem; }}
.achados-grid {{ display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:.55rem 1.8rem; }}
@media (max-width: 900px) {{ .achados-grid {{ grid-template-columns: 1fr; }} }}
.achado {{ display:flex; gap:.65rem; font-size:.9rem; color:{INK_2}; line-height:1.45; }}
.achado b {{ color:{INK}; font-weight:600; }}
.achado-n {{ flex:none; width:22px; height:22px; border-radius:6px; background:#eaf1fb; color:{MARINHO};
            font-size:.75rem; font-weight:700; display:flex; align-items:center; justify-content:center; margin-top:1px; }}

/* ---------- títulos de gráfico (a conclusão em primeiro lugar) */
.sec {{ margin: .2rem 0 .5rem; }}
.sec-t {{ font-size:1.02rem; font-weight:650; color:{INK}; letter-spacing:-.01em; line-height:1.35; }}
.sec-s {{ font-size:.82rem; color:{MUTED}; margin-top:.15rem; }}
.nota {{ font-size:.78rem; color:{MUTED}; margin-top:-.2rem; }}

/* ---------- cartões nativos (st.container(border=True)) */
div[data-testid="stVerticalBlockBorderWrapper"], div[data-testid="stVerticalBlock"][class*="border"] {{
    background:{SURFACE}; border-color:{BORDER} !important; border-radius:12px !important;
}}

/* ---------- abas: sublinhado discreto */
.stTabs [data-baseweb="tab-list"] {{ gap: 1.4rem; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{ height: 44px; padding: 0 2px; background: transparent; color:{INK_2}; }}
.stTabs [data-baseweb="tab"] p {{ font-weight:500; font-size:.92rem; }}
.stTabs [aria-selected="true"] {{ box-shadow: inset 0 -2px 0 {MARINHO}; }}
.stTabs [aria-selected="true"] p {{ color:{INK}; font-weight:600; }}
.stTabs [data-baseweb="tab-highlight"] {{ display:none; }}
.stTabs [data-baseweb="tab-border"] {{ display:none; }}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: 1.3rem; }}

/* ---------- botões */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {{
    border-radius: 9px; font-weight: 600; border:1px solid {BORDER}; transition: all .15s ease;
}}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"], .stFormSubmitButton > button[data-testid="stBaseButton-primaryFormSubmit"] {{
    background:{MARINHO}; border-color:{MARINHO}; color:#fff;
}}
.stButton > button[kind="primary"] *, .stFormSubmitButton > button[kind="primary"] *,
.stButton > button[data-testid="stBaseButton-primary"] *,
.stFormSubmitButton > button[data-testid="stBaseButton-primaryFormSubmit"] * {{ color:#fff !important; }}
.stButton > button:disabled {{ opacity:.45; }}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {{
    background:#0a3163; border-color:#0a3163;
}}

/* ---------- gráficos */
div[data-testid="stPlotlyChart"] {{ background: transparent; }}

/* ---------- etapas da entrevista */
.steps {{ display:flex; gap:.6rem; margin: .2rem 0 1.3rem; flex-wrap:wrap; }}
.step {{ flex:1 1 200px; border:1px solid {BORDER}; border-radius:10px; padding:.65rem .85rem; background:{SURFACE};
        display:flex; gap:.6rem; align-items:center; }}
.step-n {{ width:24px; height:24px; border-radius:50%; font-size:.78rem; font-weight:700; display:flex;
          align-items:center; justify-content:center; background:#f1f3f5; color:{MUTED}; flex:none; }}
.step-t {{ font-size:.86rem; font-weight:600; color:{MUTED}; line-height:1.2; }}
.step-d {{ font-size:.74rem; color:{MUTED}; font-weight:400; }}
.step.ativo {{ border-color:{MARINHO}; box-shadow: 0 0 0 1px {MARINHO} inset; }}
.step.ativo .step-n {{ background:{MARINHO}; color:#fff; }}
.step.ativo .step-t {{ color:{INK}; }}
.step.feito .step-n {{ background:#e8f5e8; color:#006300; }}
.step.feito .step-t {{ color:{INK_2}; }}

.sintese {{ background:#f7f9fc; border:1px solid {BORDER}; border-left:3px solid {MARINHO}; border-radius:8px;
           padding:.8rem 1rem; font-size:.9rem; color:{INK}; line-height:1.5; margin-bottom:.6rem; }}
.sintese small {{ display:block; color:{MUTED}; margin-top:.4rem; font-size:.8rem; }}
.aviso-campos {{ background:#fff8e6; border:1px solid #f6dfa3; border-radius:8px; padding:.6rem .9rem;
                font-size:.84rem; color:#7a4b00; margin-bottom:.8rem; }}

/* ---------- "como ler este gráfico" */
.leitura {{ background:#f7f9fc; border:1px solid {BORDER}; border-radius:10px; padding:.75rem .95rem;
           margin:.35rem 0 .2rem; font-size:.86rem; color:{INK_2}; line-height:1.55; }}
.leitura-t {{ font-size:.7rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:{MARINHO};
             margin-bottom:.35rem; }}
.leitura p {{ margin:.25rem 0; font-size:.86rem; }}
.leitura ul {{ margin:.2rem 0 .2rem 1.1rem; padding:0; }}
.leitura li {{ margin:.12rem 0; font-size:.86rem; }}
.leitura b {{ color:{INK}; }}
.leitura .alerta {{ color:#7a4b00; }}
.chk {{ display:flex; gap:.55rem; align-items:flex-start; font-size:.87rem; color:{INK_2}; padding:.3rem 0;
       border-bottom:1px solid {GRID}; }}
.chk:last-child {{ border-bottom:none; }}
.chk b {{ color:{INK}; }}
.chk-i {{ flex:none; font-size:.72rem; font-weight:700; border-radius:5px; padding:1px 6px; margin-top:1px; }}
.chk-i.ok {{ background:#e8f5e8; color:#006300; }}
.chk-i.alerta {{ background:#fff4db; color:#7a4b00; }}
.chk-i.bloqueio {{ background:#fdecea; color:#b42318; }}

/* ---------- inputs */
.stTextInput input, .stNumberInput input, .stTextArea textarea, .stDateInput input {{ border-radius: 8px; }}
[data-testid="stChatMessage"] {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px; }}
details {{ border-radius:10px !important; }}
</style>
"""

# ------------------------------------------------------------------ plotly
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color=INK_2, size=12),
    margin=dict(l=8, r=8, t=36, b=8),
    xaxis=dict(gridcolor=GRID, zeroline=False, linecolor="#d5d9e0", tickfont=dict(color=MUTED), title=None,
               showgrid=False, automargin=True),
    yaxis=dict(gridcolor=GRID, zeroline=False, linecolor="#d5d9e0", tickfont=dict(color=MUTED), title=None,
               automargin=True),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None,
                font=dict(color=INK_2, size=12), bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font=dict(color=INK, family="Inter, sans-serif", size=12)),
    colorway=list(COR_MOTIVO.values()),
    barcornerradius=4,
    bargap=0.28,
)
PALETA_CORES = list(COR_MOTIVO.values())


def tema(fig: go.Figure, altura: int = 340, **extra) -> go.Figure:
    """Aplica o tema padrão; os títulos ficam fora do gráfico (ver `secao`)."""
    fig.update_layout(**PLOTLY_THEME, height=altura, title=None)
    if extra:
        fig.update_layout(**extra)
    return fig


CONFIG_PLOTLY = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


# ------------------------------------------------------------------ componentes HTML
def secao(titulo: str, subtitulo: str = "") -> str:
    sub = f'<div class="sec-s">{subtitulo}</div>' if subtitulo else ""
    return f'<div class="sec"><div class="sec-t">{titulo}</div>{sub}</div>'


def leitura(*paragrafos: str) -> str:
    """Caixa "Como ler este gráfico". Cada argumento é um parágrafo (HTML simples) ou uma lista (<ul>)."""
    corpo = "".join(p if p.lstrip().startswith("<ul") else f"<p>{p}</p>" for p in paragrafos)
    return f'<div class="leitura"><div class="leitura-t">Como ler este gráfico</div>{corpo}</div>'


def itens(*linhas: str) -> str:
    """Lista <ul> para usar dentro de `leitura`."""
    return "<ul>" + "".join(f"<li>{l}</li>" for l in linhas) + "</ul>"


def checagens(lista: list[tuple[str, str]]) -> str:
    """Lista de verificações da base. lista: (nível ok|alerta|bloqueio, texto HTML)."""
    rotulo = {"ok": "OK", "alerta": "ATENÇÃO", "bloqueio": "BLOQUEIO"}
    return "".join(f'<div class="chk"><span class="chk-i {n}">{rotulo[n]}</span><span>{t}</span></div>'
                   for n, t in lista)


def kpi(label: str, valor, contexto: str = "", delta: str | None = None, tom: str = "neutro",
        pequeno: bool = False) -> str:
    d = f'<span class="delta {tom}">{delta}</span>' if delta else ""
    cls = "kpi-value small" if pequeno else "kpi-value"
    return (f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="{cls}">{valor}</div><div class="kpi-ctx">{d}{contexto}</div></div>')


def achados(itens: list[str]) -> str:
    li = "".join(f'<div class="achado"><span class="achado-n">{i}</span><span>{t}</span></div>'
                 for i, t in enumerate(itens, 1))
    return f'<div class="achados"><div class="achados-tit">Principais achados do recorte</div><div class="achados-grid">{li}</div></div>'


def barra_origem(n_real: int, n_sint: int) -> str:
    total = max(n_real + n_sint, 1)
    pr = n_real / total * 100
    return f"""
    <div class="origem">
      <div class="origem-txt">
        <b>Leia com cautela:</b> {pr:.0f}% dos registros deste recorte são entrevistas <b>reais</b>;
        os demais são <b>sintéticos</b> (gerados por Gaussian Copula para ampliar a amostra).
        Para conclusões institucionais, filtre por <i>Somente reais</i>.
      </div>
      <div class="origem-bar">
        <div class="origem-track">
          <div style="width:{pr:.1f}%; background:{MARINHO};"></div>
          <div style="flex:1; background:#d9dde3;"></div>
        </div>
        <div class="origem-leg"><span>■ Reais: {n_real}</span><span>Sintéticos: {n_sint}</span></div>
      </div>
    </div>"""


def etapas(atual: int) -> str:
    passos = [("Enviar entrevista", "PDF, DOCX, TXT ou texto"),
              ("Nemotron extrai os campos", "Leitura e classificação"),
              ("Revisar e gravar", "Você confirma antes de salvar")]
    html = ""
    for i, (t, d) in enumerate(passos, 1):
        cls = "ativo" if i == atual else ("feito" if i < atual else "")
        n = "✓" if i < atual else str(i)
        html += (f'<div class="step {cls}"><span class="step-n">{n}</span>'
                 f'<div><div class="step-t">{t}</div><div class="step-d">{d}</div></div></div>')
    return f'<div class="steps">{html}</div>'


def fonte_status(itens: list[tuple[str, str, str]]) -> str:
    """itens: (estado ok|warn|off, título, detalhe)."""
    linhas = ""
    for estado, titulo, det in itens:
        linhas += f'<div class="fonte-linha"><span class="dot {estado}"></span>{escape(titulo)}</div>'
        if det:
            linhas += f'<div class="fonte-det">{escape(det)}</div>'
    return f'<div class="fonte-card">{linhas}</div>'
