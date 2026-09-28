# Evasão de Militares — Painel de Inteligência sobre Desligamentos a Pedido

**Projeto Integrador I · Técnicas e Ferramentas de Visualização de Dados · MBA FGV/EMAP**

Produto analítico que transforma **entrevistas de desligamento em texto livre** de militares em **dados estruturados** e em um **painel interativo** que responde por que os militares
estão deixando a instituição, quem sai e quando. Uma LLM (NVIDIA Nemotron) lê cada entrevista bruta e propõe o
preenchimento da base; uma pessoa revisa e confirma antes de gravar.

> **A pergunta que organiza o projeto:** *que fatores levam oficiais e praças a pedir desligamento
> antes do fim da carreira — e onde a gestão deveria agir primeiro?*

---

## Sumário

1. [Problema, relevância e perguntas](#1-problema-relevância-e-perguntas)
2. [O produto: o que o painel responde](#2-o-produto-o-que-o-painel-responde)
3. [Por que uma LLM: da entrevista bruta ao dado](#3-por-que-uma-llm-da-entrevista-bruta-ao-dado)
4. [Arquitetura e estrutura de pastas](#4-arquitetura-e-estrutura-de-pastas)
5. [Como executar](#5-como-executar)
6. [Onde salvar as chaves](#6-onde-salvar-as-chaves)
7. [Configurando o banco Supabase](#7-configurando-o-banco-supabase)
8. [Como usar](#8-como-usar)
9. [Dados: origem, anonimização e dicionário](#9-dados-origem-anonimização-e-dicionário)
10. [Dados sintéticos: por que e como (Cópula Gaussiana)](#10-dados-sintéticos-por-que-e-como-cópula-gaussiana)
11. [Abordagem analítica e decisões de visualização](#11-abordagem-analítica-e-decisões-de-visualização)
12. [Problemas encontrados e o que precisa ser resolvido](#12-problemas-encontrados-e-o-que-precisa-ser-resolvido)
13. [Atendimento aos requisitos do Projeto Integrador](#13-atendimento-aos-requisitos-do-projeto-integrador)
14. [Solução de problemas](#14-solução-de-problemas)

---

## 1. Problema, relevância e perguntas

**Problema.** A instituição perde militares qualificados por desligamento voluntário antecipado. As causas estão
registradas em entrevistas de saída conduzidas pelo setor de clima organizacional, mas esses relatos
são **texto livre, dispersos em atas, relatórios e dossiês** — não há uma base consolidada que permita ver
padrões, comparar postos ou acompanhar a evolução no tempo.

**Relevância.** Cada oficial que sai leva anos de formação e experiência. Entender os motivos predominantes
(ex.: aprovação em concurso público) e os grupos mais afetados permite priorizar ações de retenção
em vez de reagir caso a caso.

**Escopo.** Desligamentos **voluntários** de oficiais e praças, 2022–2026. Ficam fora: desligamentos
involuntários (expulsão, licenciamento compulsório) e militares que completaram o ciclo de carreira.

**Perguntas que o produto responde**

| Pergunta | Onde é respondida no painel |
|---|---|
| Quantos militares estão saindo e como isso evolui? | Aba **Visão geral** |
| **Por que** saem? Quais temas aparecem nas entrevistas, e quais aparecem juntos? | Aba **Por que saem** |
| **Quem** sai? Quais postos concentram as saídas e com quais motivos? | Aba **Quem sai** |
| **Quando** saem? Há sazonalidade? Qual a ordem de grandeza esperada? | Aba **Quando saem** |
| Como incorporar uma nova entrevista à base sem digitação manual? | Aba **Nova entrevista** |

**Público e uso.** Gestores de pessoal e o setor de clima organizacional da instituição, para apoiar a
priorização de medidas de retenção.

---

## 2. O produto: o que o painel responde

Aplicação web em **Streamlit** com seis abas e uma barra lateral de filtros (origem dos registros,
período, categoria, motivo). O topo resume o recorte em **indicadores-chave** e em **achados escritos
em linguagem natural**, gerados a partir dos dados filtrados.

| Aba | Conteúdo |
|---|---|
| **Visão geral** | Série mensal com média móvel, distribuição dos motivos, volume anual |
| **Por que saem** | Composição dos motivos no tempo, % que cita cada tema, temas por ano, coocorrência de temas |
| **Quem sai** | Saídas por posto (ordem hierárquica), oficiais × praças, motivo dentro de cada posto |
| **Quando saem** | Mapa de sazonalidade mês × ano, projeção de referência (Holt) |
| **Nova entrevista** | Fluxo em 3 etapas: enviar → Nemotron extrai → revisar e gravar |
| **Consultor IA** | Parecer executivo e conversa em linguagem natural sobre o recorte filtrado |

Um aviso fixo mostra **a proporção de registros reais × sintéticos** do recorte atual (ver seção 9).

---

## 3. Por que uma LLM: da entrevista bruta ao dado

A entrevista de desligamento chega como **texto bruto** — um relato em PDF, Word ou TXT, escrito de forma
livre, com a data, o posto e os motivos misturados na narrativa. Para que ela possa ser **salva no banco e
contada nos gráficos**, é preciso convertê-la nas 12 colunas estruturadas da base. Fazer isso à mão é lento
e inconsistente entre pessoas; regras fixas (palavras-chave) falham com a variedade da linguagem.
Por isso o projeto usa uma **LLM** como etapa de extração:

```
 Entrevista bruta            Extração (LLM)                 Revisão humana            Base
 (PDF / DOCX / TXT)   ──►    NVIDIA Nemotron lê o texto ──► formulário pré-preenchido ──► Supabase
  texto livre                e devolve JSON com:            a pessoa confere e         (+ cópia no
                             data, posto, motivo,           corrige; nada é gravado    Excel local)
                             6 temas, síntese e             sem confirmação                │
                             justificativa                                                  ▼
                                                                                   Painel atualiza
```

**Como a extração é controlada** (`ia.py` → `analisar_entrevista_nemotron`)

- O prompt define as 12 colunas, a **lista fechada de motivos** e a **tabela de siglas de postos**
  (ex.: Capitão-Tenente = `CT`).
- Temperatura baixa (0,2) para respostas consistentes.
- O modelo **não inventa dados ausentes**: se a entrevista não informa a data, devolve nulo, e a tela de
  revisão avisa quais campos não estavam claros.
- O ID do registro é sempre gerado pelo sistema, nunca pelo modelo.
- A saída é validada (`normalizar_extracao`): tipos corrigidos, motivo fora da lista vira "Não Informado".
- O modelo também devolve uma **síntese** da entrevista e a **justificativa** da classificação, exibidas
  ao revisor.

> A extração é uma **proposta**, não uma decisão: a revisão humana é parte obrigatória do fluxo.
> Os riscos dessa etapa estão na [seção 12](#12-problemas-encontrados-e-o-que-precisa-ser-resolvido).

---

## 4. Arquitetura e estrutura de pastas

```
Projeto Integrador/
├── dataset_sintetico.xlsx        ← base local (reserva/backup) — precisa ficar AQUI, fora de dashboard/
└── dashboard/                    ← código do produto (execute os comandos a partir daqui)
    ├── app.py                    ← painel Streamlit (interface, gráficos, fluxo de entrevista)
    ├── dados.py                  ← leitura/gravação: Supabase (principal) e Excel (reserva)
    ├── ia.py                     ← integração com o NVIDIA Nemotron (extração e consultor)
    ├── estilos.py                ← design system: cores, tipografia, tema dos gráficos
    ├── requirements.txt          ← dependências Python
    ├── .env.example              ← modelo do .env (chave NVIDIA)
    ├── .streamlit/
    │   ├── config.toml           ← tema do Streamlit
    │   └── secrets.toml.example  ← modelo das chaves (NVIDIA e Supabase)
    └── supabase/
        ├── schema.sql            ← cria a tabela, RLS e permissões
        └── importar_excel.py     ← carrega o Excel no banco (pode rodar mais de uma vez)
```

**De onde vêm os dados**

| Situação | Fonte de leitura | Onde a nova entrevista é gravada |
|---|---|---|
| Supabase configurado e no ar | Tabela `public.desligamentos` | Supabase **e** cópia no Excel |
| Supabase não configurado | `dataset_sintetico.xlsx` | Só no Excel |
| Supabase configurado, mas fora do ar | `dataset_sintetico.xlsx` (reserva) | Não grava — o painel avisa |

A barra lateral sempre mostra qual fonte está em uso.

---

## 5. Como executar

### Pré-requisitos

- **Python 3.11 ou superior** (testado em 3.14.4)
- Opcional: chave da **NVIDIA** (para a extração de entrevistas e o consultor IA)
- Opcional: projeto no **Supabase** (para o banco de dados)

> **Sem nenhuma chave, o painel funciona**: lê o `dataset_sintetico.xlsx`, mostra todos os gráficos e
> indica "Modo local (Excel)" e "IA indisponível" na barra lateral.

### Passo a passo

**1. Baixe o projeto** mantendo a estrutura da seção 4 (o `dataset_sintetico.xlsx` fica **ao lado** da
pasta `dashboard/`, não dentro dela).

**2. Entre na pasta `dashboard`** — todos os comandos abaixo partem dela:

```bash
cd "Projeto Integrador/dashboard"
```

**3. Crie e ative um ambiente virtual**

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**4. Instale as dependências**

```bash
pip install -r requirements.txt
```

**5. (Opcional) Configure as chaves** — ver [seção 6](#6-onde-salvar-as-chaves).

**6. Rode o painel**

```bash
streamlit run app.py
```

O navegador abre em `http://localhost:8501`.

> ⚠️ Rode o comando **de dentro de `dashboard/`**. O Streamlit procura `.streamlit/secrets.toml` na pasta
> de onde é executado; rodando de outra pasta, as chaves não são encontradas e o painel cai no modo local.

---

## 6. Onde salvar as chaves

Todas as chaves ficam em **um único arquivo**, que você cria a partir do modelo:

```bash
# Windows
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
# macOS / Linux
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Depois edite `dashboard/.streamlit/secrets.toml`:

| Chave | Para que serve | Onde obter |
|---|---|---|
| `NVIDIA_API_KEY` | Extração de entrevistas e Consultor IA | [build.nvidia.com](https://build.nvidia.com) → *Get API Key* (começa com `nvapi-`) |
| `SUPABASE_URL` | Endereço do banco | Supabase → *Project Settings → Data API* (ex.: `https://abcd….supabase.co`) |
| `SUPABASE_SECRET_KEY` | Acesso do servidor ao banco | Supabase → *Project Settings → API Keys* → chave **secreta** (`sb_secret_…`) |

Alternativa só para a NVIDIA: copie `.env.example` para `.env` e preencha `NVIDIA_API_KEY=`.

**Regras de segurança**

- `secrets.toml`, `.env` e `google_credentials.json` estão no `.gitignore`. **Nunca os inclua no ZIP de
  entrega nem em repositórios.** Envie apenas os arquivos `.example`.
- Use a chave **secreta** do Supabase, nunca a publicável (`sb_publishable_…`): o banco bloqueia a chave
  publicável de propósito (ver seção 7).
- A chave secreta só é usada no servidor do Streamlit; nunca chega ao navegador de quem acessa o painel.

---

## 7. Configurando o banco Supabase

*Pule esta seção se for usar apenas o modo local.*

1. Crie um projeto em [supabase.com/dashboard](https://supabase.com/dashboard) (plano gratuito atende).
2. Abra o **SQL Editor**, cole o conteúdo de [`supabase/schema.sql`](supabase/schema.sql) e execute.
   O resultado final deve mostrar `rls = true` e `service_role:INSERT, service_role:SELECT`.
3. Preencha `SUPABASE_URL` e `SUPABASE_SECRET_KEY` no `secrets.toml` (seção 6).
4. Carregue os registros do Excel no banco:
   ```bash
   python supabase/importar_excel.py
   ```
   Saída esperada: `Excel: N linhas | banco antes: 0 | banco agora: N | novas: N` (N = linhas do Excel).
   Rodar de novo não duplica nada (IDs existentes são ignorados).
5. Rode o painel: a barra lateral deve mostrar **"Banco Supabase conectado"**.

**Modelo de acesso (por que é seguro)**

- **RLS ligado sem políticas públicas**: a chave publicável (`anon`/`authenticated`) não lê nem grava nada.
- O painel usa o papel `service_role` com **apenas `SELECT` e `INSERT`** — ele não consegue alterar nem
  apagar registros. Correções manuais são feitas pelo *Table Editor* do Supabase.
- `UNIQUE` em `id_pessoa` impede IDs duplicados; `CHECK` impede motivo fora da lista, mês fora de 1–12.

---

## 8. Como usar

**Explorar os dados.** Use os filtros da barra lateral. Para conclusões institucionais, selecione
**Origem → Reais** (ver seção 9). Todos os gráficos e os achados do topo se recalculam para o recorte.
O botão **Ver os registros do recorte em tabela** (fim da página) mostra os dados e permite baixar CSV.

**Inserir uma entrevista** (requer `NVIDIA_API_KEY`)

1. Aba **Nova entrevista** → envie um arquivo (PDF com texto, DOCX, TXT) ou cole o texto.
2. Clique em **Extrair campos com Nemotron** (leva de 30 a 40 segundos).
3. Confira a síntese, a justificativa e os campos pré-preenchidos. Se aparecer um aviso de campo
   não identificado, corrija-o.
4. **Gravar na planilha** — o registro entra no banco e os gráficos já o incluem. **Descartar** cancela.

**Consultor IA.** Gera um parecer executivo do recorte filtrado ou responde perguntas. O modelo recebe
apenas **estatísticas agregadas** do recorte, nunca os registros individuais.

---

## 9. Dados: origem, anonimização e dicionário

**Origem.** Entrevistas de desligamento de militares (2022–2026), extraídas de atas, relatórios e dossiês
internos e **anonimizadas** (nomes substituídos por ID sequencial, identificadores removidos dos textos,
códigos de quadro substituídos por `Q1`/`Q2`).

**Registros reais × sintéticos**

| Tipo | Quantidade | ID | Uso adequado |
|---|---|---|---|
| Reais | 56 (54 originais + 2 inseridos pelo fluxo do app) | `PESSOA_001…` | Base de toda conclusão qualitativa |
| Sintéticos | 446 | `SINT_001…` | Viabilizar análises que exigem volume; nunca como evidência de pessoas reais |

Os sintéticos foram gerados com **Cópula Gaussiana** (biblioteca SDV) porque a base real era pequena
demais para a análise — o porquê e o como estão na [seção 10](#10-dados-sintéticos-por-que-e-como-cópula-gaussiana).

**Dicionário de dados** (tabela `public.desligamentos` / aba `Desligamentos` do Excel)

| Coluna no banco | Coluna no Excel | Tipo | Descrição |
|---|---|---|---|
| `id` | — | bigint | Chave interna do banco |
| `id_pessoa` | `ID_Pessoa` | texto, único | `PESSOA_nnn` (real) ou `SINT_nnn` (sintético) |
| `ano`, `mes` | `Ano`, `Mes` | inteiro | Ano e mês do desligamento (base das séries temporais) |
| `data_desligamento` | `Data_Desligamento` | data | Data exata; nula quando só se sabe mês/ano |
| `posto_quadro` | `Posto_Quadro` | texto | Sigla do posto/graduação + código do quadro, ex.: `CT (Q1)`, `1TEN (Q2)`, `3SG (Q1)`. Quadros anonimizados: **Q1** = quadro principal, **Q2** = quadro complementar |
| `motivo_principal` | `Motivo_Principal` | texto (lista) | Concurso Público · Família / Vida Pessoal · Financeiro · Saúde · Outro · Não Informado |
| `menciona_*` (6 colunas) | `Menciona_*` | booleano | Se a entrevista cita o tema: concurso público, saúde, família/vida pessoal, financeiro, estudos/pós-graduação, insatisfação profissional |
| `origem_registro` | — | texto | `importacao` (veio do Excel) ou `app` (inserido pelo painel) |
| `criado_em` | — | data/hora | Momento da gravação no banco |

**Colunas derivadas pelo painel** (não gravadas): `Periodo` (1º dia do mês), `Posto` e `Quadro`
(normalizados a partir de `Posto_Quadro`), `Categoria` (Oficial / Praça), `Origem` (Real / Sintético,
pelo prefixo do ID).

---

## 10. Dados sintéticos: por que e como (Cópula Gaussiana)

### Por que foi necessário gerar dados sintéticos

A base real tem apenas **54 entrevistas** (N = 54). Com tão poucos registros, o projeto esbarrava em dois
problemas ao mesmo tempo:

| Lado A — Poder estatístico limitado | Lado B — Risco de reidentificação |
|---|---|
| Com N = 54, modelos **decoram** os dados em vez de aprender padrões (*overfitting*) e testes estatísticos perdem a sensibilidade para detectar relações verdadeiras. Cruzamentos como posto × motivo × ano ficam com células quase vazias. | Com tão poucas pessoas, cruzar variáveis categóricas (**Ano + Posto + Motivo**) pode bastar para deduzir **quem** é o militar — um risco ético e de LGPD. |

> Manter a base pequena não é sinônimo de segurança: é um gargalo analítico **e** um risco de privacidade.

A solução foi **expandir a base para N = 500** com 446 registros sintéticos, gerados por um modelo
estatístico que aprende os padrões dos 54 reais.

### O fluxo em três etapas

```
 1. A base intocável          2. O motor matemático            3. O dataset híbrido
    (dados reais)                (Cópula Gaussiana / SDV)          (N = 500)
 ┌────────────────────┐       ┌──────────────────────────┐      ┌──────────────────────────┐
 │ 54 registros reais │ ───►  │ estuda como as variáveis │ ───► │ 54 reais + 446 sintéticos│
 │ isolados; relatos e│       │ reais se relacionam entre│      │ com a mesma "estampa"    │
 │ contexto intactos  │       │ si (estrutura de         │      │ estatística dos reais    │
 └────────────────────┘       │ dependência)             │      └──────────────────────────┘
                              └──────────────────────────┘
```

### O que é a Cópula Gaussiana

Diferente de uma IA generativa "caixa-preta", a Cópula Gaussiana é um **processo estatístico
transparente**, fundamentado no **Teorema de Sklar**: qualquer distribuição conjunta de várias variáveis
pode ser separada em duas partes independentes:

1. **As margens** — o comportamento de cada variável **sozinha** (ex.: a distribuição de Ano, isolada da
   distribuição de Posto);
2. **A cópula** — a **estrutura de dependência**, ou seja, *como as variáveis se movem juntas*
   ("como os fios se cruzam"). Na Cópula Gaussiana essa estrutura é resumida em **uma única matriz de
   correlação** entre as variáveis.

Separar as duas partes permite modelar cada variável do jeito que ela é (inclusive categorias como posto
e motivo) e, ao mesmo tempo, preservar as relações entre elas.

### O processo em 4 passos

| Passo | O que acontece | No nosso caso |
|---|---|---|
| **1. Modelagem marginal** | Aprende a distribuição de **cada coluna de forma independente**. | Frequência de cada posto, de cada motivo, de cada ano e mês, e de cada tema mencionado. |
| **2. Transformação normal** | Mapeia os valores de cada coluna para uma **escala normal padrão contínua** (curva de Gauss). Categorias viram faixas numéricas proporcionais à sua frequência. | "CT", "Concurso Público", "2024" etc. passam a ser pontos numa mesma escala comparável. |
| **3. Matriz de covariância global** | Calcula **uma matriz única** que captura como todas as variáveis interagem nesse espaço normal. | Ex.: quanto "posto CT" anda junto com "motivo Concurso Público" e com "menciona Estudos". |
| **4. Amostragem inversa** | Sorteia **novos pontos** a partir dessa matriz e aplica a **transformação reversa** para voltar ao formato original. | Cada ponto sorteado vira um novo registro com Posto, Motivo, Ano, Mês e temas realistas. |

Em notação:

```
Sklar:        F(x₁, …, xₙ) = C( F₁(x₁), …, Fₙ(xₙ) )
Passo 2:      zᵢ = Φ⁻¹( Fᵢ(xᵢ) )             → cada coluna na escala normal
Passo 3:      Σ = correlação entre os zᵢ      → a "cópula"
Passo 4:      z* ~ Normal(0, Σ)  →  xᵢ* = Fᵢ⁻¹( Φ(zᵢ*) )   → novo registro sintético
```

(`Fᵢ` = distribuição da coluna i; `Φ` = distribuição normal padrão.)

**Implementação:** `GaussianCopulaSynthesizer` da biblioteca **SDV — Synthetic Data Vault** (código aberto,
[sdv.dev](https://sdv.dev)). Variáveis modeladas: Ano, Mês, Posto, Quadro, Motivo Principal e as 6 colunas
`Menciona_*`.

### Por que Cópula Gaussiana e não outra técnica

| Tecnologia | Volume de dados exigido | Risco de inventar padrões | Interpretabilidade | Velocidade |
|---|---|---|---|---|
| Redes neurais (GANs / TVAE) | Alto | Médio (*mode collapse*) | Baixa (caixa-preta) | Lenta |
| Large Language Models (LLMs) | Alto | Alto | Baixa | Muito lenta |
| **Cópula Gaussiana (escolhida)** | **Adequada a tabelas pequenas (N = 54)** | **Baixo** — só recombina padrões observados, não inventa texto nem categorias | **Alta** (a matriz pode ser inspecionada) | **Muito rápida** |

Com 54 registros, GANs e LLMs não teriam dados suficientes para aprender e seriam difíceis de explicar;
a cópula é o modelo mais simples que resolve o problema — coerente com a orientação da disciplina de não
acrescentar complexidade artificial.

### Preservação das correlações temáticas

A expansão **não é aleatória**: se, nos 54 reais, um posto aparece associado a um motivo em certo período,
essa associação se reflete **proporcionalmente** nos 446 sintéticos. As colunas de tema continuam
"amarradas" ao perfil (posto, ano) correspondente, e é isso que permite analisar coocorrência de temas e
motivo × posto com volume suficiente.

### Resguardo ético: o texto nunca é sintetizado

Há uma separação rígida entre o **estruturado** e o **não estruturado**:

- Registros **reais** mantêm o relato da entrevista (anonimizado) na base de trabalho do grupo — o relato
  não foi trazido para a tabela do painel (pendência A3 da seção 12).
- Registros **sintéticos** ficam **sem relato**: nenhuma fala foi inventada ou atribuída a pessoas
  fictícias. A síntese é estritamente **estrutural-quantitativa**.
- A **data exata** dentro do mês não é modelada: o dia é sorteado, sem significado analítico.

### Transparência e regra de uso

- **Todo registro é identificável como real ou sintético**: prefixo do ID (`PESSOA_` × `SINT_`),
  filtro **Origem** no painel e aviso permanente da proporção no topo.
- A aba `Metodologia_Dados_Sinteticos` do Excel documenta a técnica e seus limites.
- **Regra de citação:** citações diretas e conclusões qualitativas vêm **exclusivamente dos 54 reais**.
  O conjunto de 500 serve ao mapeamento de tendências, aos painéis e a testes de hipótese.

### Utilidade × privacidade

Dados sintéticos funcionam como uma **tecnologia de reforço de privacidade (PET)**: o modelo introduz
variação estruturada que dificulta a reidentificação 1-para-1 dos 54 militares reais, mantendo a
tendência macro da população. O objetivo é o ponto de equilíbrio entre **utilidade analítica** e
**privacidade (LGPD)**.

### Como provar a qualidade (framework multimétrica)

| Métrica | O que verifica |
|---|---|
| **Distância** (ex.: RMSE) | Proximidade de escala entre variáveis reais e sintéticas |
| **Correlação** (ex.: índice de Jaccard) | Se a coocorrência de categorias (posto × motivo) foi preservada |
| **Similaridade** | Integridade de cada distribuição, coluna a coluna |
| **Distribuição multivariada** (ex.: Wasserstein) | A saúde estatística do conjunto como um todo |

### Cuidados na interpretação

Para a leitura correta dos resultados, estes pontos devem acompanhar a técnica:

- **A amostragem é aleatória, não determinística.** A cópula não inventa texto nem categorias novas, mas
  pode gerar **combinações pouco plausíveis**. Na base atual isso aparece como **37 registros com data
  futura** e **concentração dos sintéticos em 2025–2026** (pendências B2 e B3 da seção 12). Fixar a
  semente aleatória torna a geração reprodutível.
- **As relações são preservadas de forma aproximada**, não exata: a cópula captura correlações de pares
  entre variáveis; relações mais complexas (ex.: um efeito que só existe para certo posto *em certo ano*)
  podem se diluir.
- **O risco de reidentificação é reduzido, não eliminado**: um registro sintético pode coincidir com a
  combinação de um real. O controle principal continua sendo a anonimização dos reais.
- **As métricas de qualidade acima ainda precisam ser calculadas e anexadas** (ex.: `QualityReport` do
  SDV, comparando real × sintético) — ver pendência B7 na seção 12.

---

## 11. Abordagem analítica e decisões de visualização

**Métodos**

- Contagens e proporções por período, motivo, posto e tema; meses sem registro contam como zero.
- **Média móvel** (janela ajustável) para separar tendência de ruído mensal.
- **Coocorrência de temas**: % de entrevistados que citam dois temas ao mesmo tempo.
- **Motivo × posto normalizado por linha**: compara postos de tamanhos diferentes.
- **Projeção de Holt** (tendência amortecida, `statsmodels`) com faixa de ~95%, apresentada como
  **referência de ordem de grandeza**, não como previsão (ver seção 12).

**Decisões de visualização**

- **Títulos que dizem a conclusão** ("Capitão-Tenente é o posto com mais desligamentos"), não só o nome
  do gráfico; subtítulo explica a métrica.
- **Cor fixa por entidade**: cada motivo tem sempre a mesma cor em todos os gráficos. Paleta categórica
  validada para daltonismo (separação ΔE ≥ 8 entre cores vizinhas); escala sequencial de um só tom.
- Grade e eixos discretos, rótulos diretos só onde importam, tabela de dados disponível como alternativa
  acessível às cores.
- Aviso permanente da proporção real × sintético, para o leitor nunca perder de vista a limitação.

**Principais achados (somente registros reais, n = 56)**

- **Concurso Público** é o motivo principal em 36 casos (64%); **Família / Vida Pessoal** vem em seguida (10 casos, 18%).
- **Capitão-Tenente (CT)** é o posto com mais saídas (16), seguido de 1TEN (13) e 2TEN (12):
  a evasão se concentra em oficiais subalternos e intermediários.

---

## 12. Problemas encontrados e o que precisa ser resolvido

Pendências conhecidas, em ordem de prioridade. Cada uma traz o impacto e a solução proposta.

### A. Extração da entrevista bruta pela LLM (etapa crítica do fluxo)

| # | Problema | Impacto | Solução proposta |
|---|---|---|---|
| A1 | **A precisão da extração não foi medida.** Não existe um conjunto de entrevistas com classificação de referência para comparar com a saída do modelo. Em teste, o modelo classificou "Capitão-Tenente" como `1TEN`; o prompt foi corrigido, mas outros erros podem existir. | Erros silenciosos entram na base se o revisor não perceber. | Montar um **conjunto de validação**: as 54 entrevistas reais classificadas por uma pessoa, comparadas campo a campo com a saída do modelo (acerto por campo, matriz de confusão de motivos). Repetir a cada mudança de prompt ou modelo. |
| A2 | **O texto completo da entrevista é enviado a uma API externa** (NVIDIA, fora do Brasil). | Risco de exposição de dados pessoais e institucionais; exige base legal (LGPD) e autorização da instituição. | Anonimizar **antes** do envio (remover nomes, matrícula/identidade militar, unidade e lotação — por regras e/ou NER), obter autorização formal, ou rodar o modelo localmente (NVIDIA NIM auto-hospedado ou modelo aberto). |
| A3 | **O texto bruto e os metadados da extração não são guardados.** A tabela só recebe as 12 colunas finais; a coluna `Relato_Entrevista` citada na metodologia não existe na base. | Impossível auditar uma classificação, reprocessar com um prompt melhor ou medir o quanto o revisor corrige o modelo. | Adicionar ao banco: relato anonimizado, modelo e versão do prompt, saída original da LLM, campos alterados na revisão, revisor e data. |
| A4 | **O motivo principal é único**, mas as entrevistas costumam ser multicausais (ex.: concurso + família + remuneração). | A escolha do "principal" é interpretativa e pode variar entre revisores e execuções. | Documentar o critério de desempate (já no prompt: concurso aprovado prevalece) e usar os 6 temas para a leitura multicausal. |
| A5 | **PDF escaneado (imagem) não tem texto extraível.** | A extração falha; hoje o usuário precisa colar o texto manualmente. | Adicionar OCR (ex.: Tesseract) para PDFs sem camada de texto. |
| A6 | **Latência de 30–40 s e dependência da API** (disponibilidade, cota gratuita, mudança de modelo). | Fluxo lento; indisponibilidade bloqueia a inserção. | Nova tentativa automática, processamento em lote, e fixar a versão do modelo documentada. |
| A7 | **Respostas não determinísticas**: a mesma entrevista pode gerar saídas diferentes. | Baixa reprodutibilidade. | Temperatura já reduzida a 0,2; registrar a saída original (A3) e avaliar temperatura 0. |

### B. Dados

| # | Problema | Impacto | Solução proposta |
|---|---|---|---|
| B1 | **Só 56 registros são reais; 446 (89%) são sintéticos.** | Conclusões gerais dependem de dados gerados. | Toda conclusão qualitativa deve usar o filtro **Reais**; ampliar a coleta de entrevistas reais. |
| B2 | **Os sintéticos estão concentrados em 2025–2026** (em 2026: 189 sintéticos × 18 reais). | A tendência de alta ("▲96%" nos últimos 12 meses) e a projeção de Holt refletem mais o gerador do que a realidade. | Regerar os sintéticos respeitando a distribuição temporal dos reais, ou calcular tendência e projeção **só com reais**. |
| B3 | **37 registros sintéticos têm data futura** (após 09/2026). | Distorcem as séries se exibidos. | Já ocultados por padrão (*Opções avançadas*); corrigir na geração. |
| B4 | **A aba de metodologia não corresponde à base**: cita as colunas `Tipo_Registro` e `Relato_Entrevista`, que não existem; o tipo é inferido pelo prefixo do ID (`SINT_`). | Documentação inconsistente; a regra do prefixo é frágil. | Alinhar a metodologia à base ou criar a coluna de tipo explicitamente. |
| B5 | **Inconsistências de preenchimento**: 23 grafias diferentes de `Posto_Quadro` (ex.: `CC(Q1)`, `2 TEN (Q1)`, `1T(Q1)` — com e sem espaço, siglas abreviadas), 2 registros sem data, 2 só com mês. | Contagens erradas se não tratadas. | Já normalizado no código; padronizar na origem (o formulário de revisão já usa lista fechada). |
| B6 | **Confirmar o `PESSOA_055` e o `PESSOA_056`**: registros inseridos pelo fluxo do app; a proposta cita 54 entrevistas reais. | Podem ser registros de teste contados como reais. | O grupo deve confirmar se correspondem a entrevistas reais; se não, removê-los pelo Table Editor do Supabase e do Excel. |
| B7 | **A geração dos sintéticos não está documentada como código nem validada.** O script do SDV (com a semente aleatória) não faz parte da entrega, e as métricas de qualidade da seção 10 não foram calculadas. | Não é possível reproduzir a base nem provar que os sintéticos preservam os padrões dos reais. | Incluir o script de geração com semente fixa e anexar o relatório de qualidade do SDV (`QualityReport`: similaridade por coluna e por pares de colunas). |

### C. Infraestrutura e segurança

| # | Problema | Impacto | Solução proposta |
|---|---|---|---|
| C1 | **O painel não tem login.** | Se publicado na internet, qualquer pessoa com o link vê os dados e pode inserir registros. | Usar apenas localmente, ou adicionar autenticação (ex.: `st.login` com provedor OIDC) antes de qualquer publicação. |
| C2 | **O plano gratuito do Supabase pausa o projeto após ~7 dias sem uso.** | O painel cai no Excel e não grava novas entrevistas. | Reativar o projeto no dashboard do Supabase antes de usar/avaliar; o modo local garante a leitura. |
| C3 | **Chaves reais na pasta de trabalho** (`.env`, `secrets.toml`, `google_credentials.json`). | Vazamento se a pasta for compactada inteira para entrega. | Montar o ZIP sem esses arquivos (só os `.example`); `google_credentials.json` é sobra da versão com Google Sheets e pode ser apagado. |
| C4 | **O Excel fica fora da pasta do código** (`../dataset_sintetico.xlsx`). | Quem baixar só a pasta `dashboard/` fica sem a base local. | Manter a estrutura da seção 4 ou mover o Excel para dentro de `dashboard/data/`. |
| C5 | **Não há testes automatizados.** | Mudanças podem quebrar o fluxo sem aviso. | Testes para `normalizar_extracao`, `para_linha_db` e para o carregamento sem credenciais (`streamlit.testing`). |
| C6 | **Sobras da tentativa com Google Sheets** no Drive do grupo (cópia nativa da planilha e projeto Apps Script sem uso). | Nenhum funcional; confusão sobre qual é a base oficial. | Apagar. A base oficial é o Supabase, com o Excel como reserva. |

---

## 13. Atendimento aos requisitos do Projeto Integrador

**Entregáveis (16 de outubro)**

| Entregável exigido | Situação |
|---|---|
| Produto analítico final | ✅ Este painel (`app.py`) |
| Relatório escrito curto (problema, objetivo, dados, abordagem, achados, conclusões) | ⏳ A produzir pelo grupo — as seções 1, 3, 9, 10, 11 e 12 deste README servem de base |
| Arquivos-fonte | ✅ Código, SQL, script de importação e base Excel |
| Documentação mínima (uso, estrutura dos dados, dependências, observações) | ✅ Este README (seções 5–9, dependências abaixo, pendências na seção 12) |

**Dimensões de avaliação**

| Dimensão | Como o projeto responde |
|---|---|
| Problema e escopo | Problema definido, público identificado, escopo com o que fica fora (seção 1) |
| Dados | Origem, anonimização, real × sintético e dicionário documentados; limitações explícitas (seções 9, 10 e 12-B) |
| Análise | Perguntas ligadas a métodos e visualizações específicas (seções 1 e 11) |
| Visualização e comunicação | Títulos-conclusão, achados em linguagem natural, cores consistentes e acessíveis (seções 2 e 11) |
| Execução técnica | Extração por LLM com revisão humana, banco com integridade e controle de acesso, funcionamento sem credenciais (seções 3, 4 e 7) |
| Coerência do produto final | Da entrevista bruta à decisão: problema → perguntas → dados → análise → visualização → comunicação |

**Dependências** (versões testadas; ver `requirements.txt`)

`streamlit 1.64.0` · `plotly 7.1.0` · `pandas 2.3.3` · `openpyxl 3.1.5` · `statsmodels 0.15.0` ·
`openai 2.41.0` (cliente da API NVIDIA, compatível com OpenAI) · `python-docx 1.2.0` · `pypdf 6.10.2` ·
`supabase 2.31.0` — modelo de IA: `nvidia/nemotron-3.5-lightning-30b-a3b` via NVIDIA NIM.

---

## 14. Solução de problemas

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| Barra lateral mostra "Modo local (Excel)" mesmo com as chaves | Comando rodado fora de `dashboard/`, ou `secrets.toml` não criado | `cd dashboard` e `streamlit run app.py`; confira o nome do arquivo (sem `.example`) |
| "Supabase indisponível" | Projeto pausado (plano gratuito) ou chave errada | Reative o projeto no dashboard do Supabase; confira se a chave é a **secreta** |
| "IA indisponível" / botão de extração desabilitado | `NVIDIA_API_KEY` ausente | Preencha em `secrets.toml` ou `.env` |
| "O ID PESSOA_xxx já existe no banco" | Outra pessoa gravou um registro ao mesmo tempo | Clique em **Atualizar** e grave de novo (o próximo ID é recalculado) |
| `permission denied for table desligamentos` | `schema.sql` não executado, ou chave publicável em uso | Rode o `schema.sql` e use a chave secreta |
| "Não foi possível extrair texto do arquivo" | PDF escaneado (imagem) | Cole o texto manualmente (ver pendência A5) |
| Arquivo `dataset_sintetico.xlsx` não encontrado | Excel fora do lugar | Coloque-o na pasta acima de `dashboard/` (seção 4) |
