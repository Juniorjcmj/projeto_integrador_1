"""Carga, limpeza e gravação do dataset de desligamentos a pedido de militares.

Fonte principal: tabela `public.desligamentos` no Supabase (ver supabase/schema.sql).
Fonte reserva: `dataset_sintetico.xlsx`, usada quando o Supabase não está configurado ou não responde —
assim o painel roda para quem não tem as credenciais (ex.: avaliação da entrega).
"""
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

ARQUIVO_LOCAL = Path(__file__).resolve().parent.parent / "dataset_sintetico.xlsx"
TABELA = "desligamentos"

TEMAS = {
    "Menciona_ConcursoPublico": "Concurso Público",
    "Menciona_Saude": "Saúde",
    "Menciona_FamiliaVidaPessoal": "Família / Vida Pessoal",
    "Menciona_Financeiro": "Financeiro",
    "Menciona_EstudosPosGraduacao": "Estudos / Pós-graduação",
    "Menciona_InsatisfacaoProfissional": "Insatisfação Profissional",
}

MOTIVOS = ["Concurso Público", "Família / Vida Pessoal", "Financeiro", "Saúde", "Outro", "Não Informado"]

# Formato canônico de Posto_Quadro para o formulário de revisão.
# Quadros anonimizados: Q1 = quadro principal, Q2 = quadro complementar.
POSTOS_QUADRO = [
    "GM (Q1)", "2TEN (Q1)", "1TEN (Q1)", "CT (Q1)", "CC (Q1)", "CMG (Q1)", "3SG (Q1)",
    "2TEN (Q2)", "1TEN (Q2)", "CT (Q2)", "CC (Q2)", "CMG (Q2)", "Não Informado (Q1)",
]

NOMES_POSTOS = {
    "GM": "Guarda-Marinha", "2TEN": "Segundo-Tenente", "1TEN": "Primeiro-Tenente",
    "CT": "Capitão-Tenente", "CC": "Capitão de Corveta", "CMG": "Capitão de Mar e Guerra",
    "3SG": "Terceiro-Sargento",
}

_POSTO_ALIAS = {"1T": "1TEN", "2T": "2TEN"}
_OFICIAIS = {"GM", "2TEN", "1TEN", "CT", "CC", "CMG"}
ORDEM_POSTOS = ["GM", "2TEN", "1TEN", "CT", "CC", "CMG", "3SG"]

# Colunas da planilha (usadas no app) -> colunas da tabela no Postgres (snake_case).
COLUNAS_DB = {
    "ID_Pessoa": "id_pessoa",
    "Ano": "ano",
    "Mes": "mes",
    "Data_Desligamento": "data_desligamento",
    "Posto_Quadro": "posto_quadro",
    "Motivo_Principal": "motivo_principal",
    "Menciona_ConcursoPublico": "menciona_concurso_publico",
    "Menciona_Saude": "menciona_saude",
    "Menciona_FamiliaVidaPessoal": "menciona_familia_vida_pessoal",
    "Menciona_Financeiro": "menciona_financeiro",
    "Menciona_EstudosPosGraduacao": "menciona_estudos_pos_graduacao",
    "Menciona_InsatisfacaoProfissional": "menciona_insatisfacao_profissional",
}
COLUNAS_RAW = list(COLUNAS_DB)


# ------------------------------------------------------------------ configuração / conexão
def _config(nome: str, padrao: str | None = None) -> str | None:
    """Lê um parâmetro de .streamlit/secrets.toml ou de variável de ambiente."""
    try:
        import streamlit as st
        valor = st.secrets.get(nome)
        if valor:
            return valor
    except Exception:
        pass
    return os.environ.get(nome, padrao)


def _url_supabase() -> str | None:
    """URL base do projeto; aceita também a do endpoint REST ('https://<ref>.supabase.co/rest/v1/')."""
    from urllib.parse import urlsplit

    url = (_config("SUPABASE_URL") or "").strip()
    if not url:
        return None
    partes = urlsplit(url)
    return f"{partes.scheme}://{partes.netloc}"


def supabase_configurado() -> bool:
    return bool(_url_supabase() and _config("SUPABASE_SECRET_KEY"))


def url_painel_supabase() -> str | None:
    """Link para a tabela no dashboard do Supabase (https://<ref>.supabase.co -> project/<ref>)."""
    url = _url_supabase()
    if not url:
        return None
    ref = url.split("//")[-1].split(".")[0]
    return f"https://supabase.com/dashboard/project/{ref}/editor"


@lru_cache(maxsize=1)
def _cliente():
    """Cliente com a chave secreta — só roda no servidor (Streamlit); nunca vai para o navegador."""
    from supabase import create_client

    return create_client(_url_supabase(), _config("SUPABASE_SECRET_KEY").strip())


# ------------------------------------------------------------------ leitura
def _ler_supabase() -> pd.DataFrame:
    linhas, passo, i = [], 1000, 0
    while True:  # a Data API devolve no máximo ~1000 linhas por chamada
        lote = (_cliente().table(TABELA).select(",".join(COLUNAS_DB.values()))
                .order("id").range(i, i + passo - 1).execute().data)
        linhas += lote
        if len(lote) < passo:
            break
        i += passo
    df = pd.DataFrame(linhas, columns=list(COLUNAS_DB.values()))
    return df.rename(columns={v: k for k, v in COLUNAS_DB.items()})


def _ler_excel() -> pd.DataFrame:
    return pd.read_excel(ARQUIVO_LOCAL, sheet_name="Desligamentos")


def carregar() -> pd.DataFrame:
    """Carrega do Supabase; se não houver credenciais ou o banco falhar, usa o Excel local."""
    erro = None
    if supabase_configurado():
        try:
            df, fonte = _ler_supabase(), "Supabase"
        except Exception as e:
            erro = str(e)
    if not supabase_configurado() or erro:
        if not ARQUIVO_LOCAL.exists():
            raise RuntimeError(f"Supabase indisponível ({erro}) e arquivo local não encontrado.")
        df, fonte = _ler_excel(), f"Arquivo local ({ARQUIVO_LOCAL.name})"

    # Período: usa Ano/Mês; se faltarem, tenta a data textual ('2022-07').
    data = pd.to_datetime(df["Data_Desligamento"], errors="coerce")
    ano = pd.to_numeric(df["Ano"], errors="coerce").fillna(data.dt.year)
    mes = pd.to_numeric(df["Mes"], errors="coerce").fillna(data.dt.month)
    df["Periodo"] = pd.to_datetime({"year": ano, "month": mes, "day": 1}, errors="coerce")

    pq = df["Posto_Quadro"].map(_posto_quadro)
    df["Posto"] = pq.str[0]
    df["Quadro"] = pq.str[1]
    df["Categoria"] = df["Posto"].map(
        lambda p: "Oficial" if p in _OFICIAIS else ("Praça" if p.endswith("SG") else "Não Informado")
    )

    df["Origem"] = df["ID_Pessoa"].astype(str).str.startswith("SINT").map({True: "Sintético", False: "Real"})
    df = df.rename(columns=TEMAS)
    for col in TEMAS.values():
        df[col] = df[col].map(_para_bool).astype(bool)
    df["Motivo_Principal"] = df["Motivo_Principal"].fillna("Não Informado")
    df.attrs["fonte"] = fonte
    df.attrs["erro_fonte"] = erro
    return df


def _posto_quadro(txt):
    """'2 TEN (XX)' -> ('2TEN', 'XX'); '3°SG' -> ('3SG', 'Não Informado')."""
    if pd.isna(txt):
        return "Não Informado", "Não Informado"
    txt = str(txt).replace("°", "").strip()
    posto, _, resto = txt.partition("(")
    posto = posto.replace(" ", "").upper()
    quadro = resto.rstrip(")").strip() or "Não Informado"
    posto = _POSTO_ALIAS.get(posto, posto)
    if posto.startswith("NÃO"):
        posto = "Não Informado"
    return posto, quadro


def _para_bool(v) -> bool:
    """Aceita True/False, 1/0 e textos ('TRUE', 'sim', 'false')."""
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "sim", "s", "yes", "verdadeiro"}
    return bool(v) if pd.notna(v) else False


def serie_mensal(df: pd.DataFrame, por=None) -> pd.DataFrame:
    """Contagem mensal (meses vazios = 0), opcionalmente quebrada por uma coluna."""
    d = df.dropna(subset=["Periodo"])
    if d.empty:
        return pd.DataFrame(columns=["Periodo", "Qtde"])
    meses = pd.date_range(d["Periodo"].min(), d["Periodo"].max(), freq="MS")
    if por is None:
        s = d.groupby("Periodo").size().reindex(meses, fill_value=0)
        return s.rename_axis("Periodo").reset_index(name="Qtde")
    t = d.groupby(["Periodo", por]).size().unstack(fill_value=0).reindex(meses, fill_value=0)
    return t.rename_axis("Periodo").reset_index().melt("Periodo", var_name=por, value_name="Qtde")


def proximo_id_pessoa(df: pd.DataFrame) -> str:
    """Calcula o próximo ID sequencial de registro real (ex: PESSOA_055)."""
    if df.empty or "ID_Pessoa" not in df.columns:
        return "PESSOA_001"
    numeros = [int(p.split("_")[1]) for p in df["ID_Pessoa"].astype(str)
               if p.startswith("PESSOA_") and p.split("_")[1].isdigit()]
    return f"PESSOA_{(max(numeros) + 1) if numeros else 1:03d}"


# ------------------------------------------------------------------ gravação
def para_linha_db(registro: dict, origem: str = "app") -> dict:
    """Converte um registro no formato da planilha para uma linha da tabela."""
    linha = {}
    for col, col_db in COLUNAS_DB.items():
        v = registro.get(col)
        if col.startswith("Menciona_"):  # tema ausente = não mencionado (coluna é NOT NULL)
            v = _para_bool(v)
        elif v is None or (not isinstance(v, (list, dict)) and pd.isna(v)):
            v = None
        elif col in ("Ano", "Mes"):
            v = int(v)
        elif col == "Data_Desligamento":
            d = pd.to_datetime(str(v), format="%Y-%m-%d", errors="coerce")  # '2022-07' vira nulo
            v = d.date().isoformat() if pd.notna(d) else None
        linha[col_db] = v
    linha["origem_registro"] = origem
    return linha


def _gravar_excel(registro: dict) -> str | None:
    """Acrescenta a linha no Excel local. Devolve mensagem de erro ou None."""
    import openpyxl

    if not ARQUIVO_LOCAL.exists():
        return f"Arquivo local {ARQUIVO_LOCAL.name} não encontrado."
    wb = openpyxl.load_workbook(ARQUIVO_LOCAL)
    ws = wb["Desligamentos"]
    if str(registro.get("ID_Pessoa")) in {str(c.value) for c in ws["A"][1:]}:
        return f"O ID {registro.get('ID_Pessoa')} já existe no Excel local."
    ws.append([registro.get(col) for col in COLUNAS_RAW])
    wb.save(ARQUIVO_LOCAL)
    return None


def gravar_registro(registro: dict) -> dict:
    """Grava no Supabase (fonte oficial) e espelha no Excel local como backup.

    Sem Supabase configurado, grava só no Excel. Devolve {ok, destino, mensagem, aviso}.
    """
    if not supabase_configurado():
        erro = _gravar_excel(registro)
        return {"ok": erro is None, "destino": "excel", "mensagem": erro or "", "aviso": ""}

    try:
        _cliente().table(TABELA).insert(para_linha_db(registro)).execute()
    except Exception as e:
        codigo = getattr(e, "code", "")
        msg = (f"O ID {registro.get('ID_Pessoa')} já existe no banco. Atualize o painel e tente de novo."
               if codigo == "23505" else f"Falha ao gravar no Supabase: {getattr(e, 'message', e)}")
        return {"ok": False, "destino": "supabase", "mensagem": msg, "aviso": ""}

    try:
        aviso = _gravar_excel(registro) or ""
    except Exception as e:  # o backup nunca deve derrubar uma gravação que já deu certo
        aviso = f"Backup no Excel falhou: {e}"
    return {"ok": True, "destino": "supabase", "mensagem": "", "aviso": aviso}
