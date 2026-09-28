"""Integração com a API NVIDIA NIM — Modelo Nemotron.

Permite gerar análises executivas e responder dúvidas em linguagem natural
com base nos dados filtrados do dashboard de evasão.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Generator
import pandas as pd
from openai import OpenAI

DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


def ia_configurada() -> bool:
    """Indica se há chave da NVIDIA disponível (sem chamar a API)."""
    try:
        get_nvidia_client()
        return True
    except ValueError:
        return False


def get_nvidia_client(api_key: str | None = None) -> OpenAI:
    """Cria e retorna o cliente OpenAI configurado para o endpoint da NVIDIA."""
    key = api_key or os.environ.get("NVIDIA_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("NVIDIA_API_KEY")
        except Exception:
            pass

    if not key:
        env_file = Path(__file__).resolve().parent / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("NVIDIA_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

    if not key:
        raise ValueError(
            "NVIDIA_API_KEY não configurada. Defina-a em .streamlit/secrets.toml ou no .env (veja o README)."
        )
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=key)


def get_system_prompt(df: pd.DataFrame) -> str:
    """Retorna o prompt de sistema especializado com o contexto atualizado dos dados."""
    resumo = resumir_dados_para_ia(df)
    return f"""Você é o Consultor Sênior de Inteligência Estratégica e People Analytics de uma instituição militar.
Refira-se sempre aos "militares" e à "instituição", sem nomear Força, corpo ou organização militar específica.
Sua missão é assessorar os decisores militares na compreensão do fenômeno de evasão / desligamento a pedido.

DIRETRIZES FUNDAMENTAIS:
1. Rigor Metodológico: Registros 'Sintéticos' foram gerados por modelos estocásticos (Gaussian Copula); sempre sinalize quando uma conclusão puder ser enviesada por dados sintéticos vs. dados reais.
2. Foco Estratégico: Avalie os impactos da perda de capital humano qualificado (ex: Oficiais Subalternos e Intermediários como 1TEN e CT, e Praças especializadas como 3SG).
3. Linguagem: Responda em Português do Brasil com postura executiva, clara, analítica e estruturada (use tópicos, negrito e destaques).
4. Praticidade: Forneça recomendações acionáveis e realistas dentro do contexto militar/estatal.

{resumo}
"""


def resumir_dados_para_ia(df: pd.DataFrame) -> str:
    """Cria um resumo estatístico e contextual do DataFrame filtrado para o prompt do modelo."""
    if df.empty:
        return "Nenhum registro encontrado com os filtros atuais."

    total = len(df)
    n_real = int((df["Origem"] == "Real").sum())
    n_sint = total - n_real

    # Datas
    dfp = df.dropna(subset=["Periodo"])
    periodo_str = "Indefinido"
    if not dfp.empty:
        p_min = dfp["Periodo"].min().strftime("%m/%Y")
        p_max = dfp["Periodo"].max().strftime("%m/%Y")
        periodo_str = f"{p_min} a {p_max}"

    # Distribuição de categorias e postos
    cat_counts = df["Categoria"].value_counts().to_dict()
    posto_counts = df["Posto"].value_counts().head(5).to_dict()

    # Motivos principais
    motivos_counts = df["Motivo_Principal"].value_counts().to_dict()

    # Temas mencionados
    colunas_temas = [
        "Concurso Público",
        "Saúde",
        "Família / Vida Pessoal",
        "Financeiro",
        "Estudos / Pós-graduação",
        "Insatisfação Profissional",
    ]
    temas_stats = {}
    for col in colunas_temas:
        if col in df.columns:
            qtd = int(df[col].astype(bool).sum())
            pct = (qtd / total * 100) if total > 0 else 0
            temas_stats[col] = f"{qtd} ({pct:.1f}%)"

    resumo = f"""
--- RECORTE ATUAL DOS DADOS (APÓS FILTROS DO USUÁRIO) ---
- Período avaliado: {periodo_str}
- Total de registros: {total} (Reais: {n_real} | Sintéticos: {n_sint})
- Distribuição por Categoria: {cat_counts}
- Top 5 Postos/Graduações com mais desligamentos: {posto_counts}
- Motivos Principais apontados: {motivos_counts}
- Temas mencionados nas entrevistas de saída:
{chr(10).join(f"  * {t}: {v}" for t, v in temas_stats.items())}
--------------------------------------------------------
"""
    return resumo.strip()


def stream_nemotron(
    messages: list[dict],
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    enable_thinking: bool = True,
    reasoning_budget: int | None = None,
) -> Generator[tuple[str, str], None, None]:
    """Chama a API do Nemotron em modo streaming.

    Gera tuplas (tipo, texto), onde tipo pode ser 'reasoning' (raciocínio interno)
    ou 'content' (resposta final ao usuário).
    """
    client = get_nvidia_client(api_key)

    extra_body = {}
    if enable_thinking:
        budget = reasoning_budget if reasoning_budget is not None else min(max_tokens, 1024)
        extra_body = {
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_budget": budget,
        }

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
        extra_body=extra_body if extra_body else None,
        stream=True,
    )

    for chunk in completion:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        # Raciocínio (thinking)
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            yield ("reasoning", reasoning)

        # Conteúdo final
        content = delta.content
        if content is not None:
            yield ("content", content)


def extrair_texto_arquivo(uploaded_file) -> str:
    """Extrai texto de arquivos .txt, .docx ou .pdf enviados no Streamlit."""
    nome = uploaded_file.name.lower()
    conteudo_bytes = uploaded_file.getvalue()

    if nome.endswith(".txt"):
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                return conteudo_bytes.decode(enc)
            except UnicodeDecodeError:
                continue
        return conteudo_bytes.decode("utf-8", errors="ignore")

    elif nome.endswith(".docx"):
        import io
        import docx

        doc = docx.Document(io.BytesIO(conteudo_bytes))
        paragrafos = [p.text for p in doc.paragraphs if p.text.strip()]
        for tab in doc.tables:
            for row in tab.rows:
                linha = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                if linha:
                    paragrafos.append(linha)
        return "\n".join(paragrafos)

    elif nome.endswith(".pdf"):
        import io
        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(conteudo_bytes))
        paginas = []
        for p in reader.pages:
            t = p.extract_text()
            if t:
                paginas.append(t)
        return "\n".join(paginas)

    return ""


def analisar_entrevista_nemotron(
    texto: str,
    proximo_id: str = "PESSOA_055",
    api_key: str | None = None,
) -> tuple[dict, str, str]:
    """Usa o Nemotron 3.5 para extrair os 12 campos estruturados da entrevista."""
    import json
    import re

    from dados import POSTOS_QUADRO

    quadros = sorted({p.partition("(")[2].rstrip(")") for p in POSTOS_QUADRO} - {""})
    lista_quadros = ", ".join('"' + q + '"' for q in quadros)
    exemplos_posto = ", ".join(POSTOS_QUADRO[:3])
    prompt_instrucao = f"""Você é o analista sênior de People Analytics de uma instituição militar.
Sua missão é ler o relatório de entrevista de desligamento a pedido e preencher com precisão os 12 campos estruturados da planilha.

CAMPOS A PREENCHER NO JSON:
- "ID_Pessoa": "{proximo_id}"
- "Ano": número inteiro do ano do desligamento (ex: 2024; se não constar no relatório, use null)
- "Mes": número inteiro do mês (1 a 12; se não constar, use null)
- "Data_Desligamento": data no formato "AAAA-MM-DD" (se o dia exato não constar, use "AAAA-MM-01"; se nem mês/ano constarem, use null)
- "Posto_Quadro": string normalizada "SIGLA (QUADRO)", sem espaço na sigla. Siglas obrigatórias:
  Guarda-Marinha = GM; Segundo-Tenente = 2TEN; Primeiro-Tenente = 1TEN; Capitão-Tenente = CT;
  Capitão de Corveta = CC; Capitão de Mar e Guerra = CMG; Terceiro-Sargento = 3SG.
  QUADRO: um destes códigos: {lista_quadros}. Use Q2 quando o relato indicar Quadro Complementar;
  Q1 nos demais casos. Formatos válidos: {exemplos_posto}.
  Atenção: Capitão-Tenente é CT, NÃO é 1TEN.
- "Motivo_Principal": motivo preponderante, escolha exatamente um entre:
  ["Concurso Público", "Família / Vida Pessoal", "Financeiro", "Saúde", "Outro", "Não Informado"]
  (Atenção: Se o militar estudou e foi aprovado em concurso público para assumir cargo civil, classifique como "Concurso Público", mesmo que haja insatisfação como gatilho de fundo).
- Indicadores booleanos dos temas mencionados na entrevista (true se o militar ou o relato abordou o tema, false se não):
  "Menciona_ConcursoPublico": bool (ex: aprovação em tribunal, receita, polícia, etc.)
  "Menciona_Saude": bool (ex: lesão, estresse, saúde familiar ou própria)
  "Menciona_FamiliaVidaPessoal": bool (ex: casado, filhos, transferência familiar, tempo com família)
  "Menciona_Financeiro": bool (ex: melhor remuneração externa, salário, condições financeiras)
  "Menciona_EstudosPosGraduacao": bool (ex: estudos para concursos, cursos de pós-graduação, aperfeiçoamento)
  "Menciona_InsatisfacaoProfissional": bool (ex: queixas com liderança/comando, falta de perspectiva, desmotivação com a carreira)
- "Justificativa_Classificacao": justificativa técnica de 1 a 2 frases
- "Sintese_Entrevista": resumo conciso em 2 frases sobre a trajetória e motivo de saída do militar

Retorne o JSON em bloco:
```json
{{ ... }}
```

RELATÓRIO DA ENTREVISTA:
{texto}
"""
    messages = [
        {
            "role": "system",
            "content": "Você é um assistente de extração precisa de dados estruturados para People Analytics militar. Retorne estritamente o JSON solicitado.",
        },
        {"role": "user", "content": prompt_instrucao},
    ]

    thinking = ""
    content = ""
    for tipo, chunk in stream_nemotron(
        messages,
        api_key=api_key,
        temperature=0.2,  # extração: queremos consistência, não criatividade
        max_tokens=4096,
        enable_thinking=True,
        reasoning_budget=1024,
    ):
        if tipo == "reasoning":
            thinking += chunk
        elif tipo == "content":
            content += chunk

    # Tenta extrair JSON de content ou thinking
    texto_busca = content if "{" in content else (thinking + "\n" + content)
    match = re.search(r"\{[\s\S]*\}", texto_busca)
    if not match:
        raise ValueError("O modelo não retornou uma estrutura JSON válida. Tente novamente.")

    dados = json.loads(match.group(0))
    return normalizar_extracao(dados, proximo_id), thinking, content


CAMPOS_TEMAS = [
    "Menciona_ConcursoPublico",
    "Menciona_Saude",
    "Menciona_FamiliaVidaPessoal",
    "Menciona_Financeiro",
    "Menciona_EstudosPosGraduacao",
    "Menciona_InsatisfacaoProfissional",
]


def normalizar_extracao(dados: dict, proximo_id: str) -> dict:
    """Coage tipos do JSON do modelo e registra quais campos ele não conseguiu inferir."""
    from datetime import date

    from dados import MOTIVOS, POSTOS_QUADRO

    faltantes = []

    def _int(v):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None

    dados["ID_Pessoa"] = proximo_id  # o ID é sempre do sistema, nunca do modelo

    data = pd.to_datetime(dados.get("Data_Desligamento"), errors="coerce")
    ano, mes = _int(dados.get("Ano")), _int(dados.get("Mes"))
    if pd.isna(data):
        if ano and mes and 1 <= mes <= 12:
            data = pd.Timestamp(year=ano, month=mes, day=1)
        else:
            data = pd.Timestamp(date.today().replace(day=1))
            faltantes.append("Data_Desligamento")
    dados["Data_Desligamento"] = data.date()
    dados["Ano"], dados["Mes"] = data.year, data.month

    if dados.get("Motivo_Principal") not in MOTIVOS:
        faltantes.append("Motivo_Principal")
        dados["Motivo_Principal"] = "Não Informado"

    if not str(dados.get("Posto_Quadro") or "").strip():
        faltantes.append("Posto_Quadro")
        dados["Posto_Quadro"] = next(p for p in POSTOS_QUADRO if p.startswith("Não Informado"))

    for campo in CAMPOS_TEMAS:
        v = dados.get(campo)
        dados[campo] = v.strip().lower() in {"true", "sim", "1"} if isinstance(v, str) else bool(v)

    dados["_faltantes"] = faltantes
    return dados

