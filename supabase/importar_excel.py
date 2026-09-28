"""Importa dataset_sintetico.xlsx para a tabela public.desligamentos do Supabase.

Uso (na pasta dashboard/, depois de rodar supabase/schema.sql no SQL Editor):
    python supabase/importar_excel.py

Lê SUPABASE_URL e SUPABASE_SECRET_KEY de .streamlit/secrets.toml (ou de variáveis de ambiente).
É seguro rodar mais de uma vez: IDs que já existem no banco são ignorados.
"""
import os
import sys
import tomllib
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

segredos = RAIZ / ".streamlit" / "secrets.toml"
if segredos.exists():
    for k, v in tomllib.loads(segredos.read_text(encoding="utf-8")).items():
        os.environ.setdefault(k, str(v))

import pandas as pd  # noqa: E402
from postgrest.types import ReturnMethod  # noqa: E402

import dados  # noqa: E402


def main():
    if not dados.supabase_configurado():
        sys.exit("Defina SUPABASE_URL e SUPABASE_SECRET_KEY em .streamlit/secrets.toml.")

    excel = pd.read_excel(dados.ARQUIVO_LOCAL, sheet_name="Desligamentos")
    linhas = [dados.para_linha_db(r, origem="importacao") for r in excel[dados.COLUNAS_RAW].to_dict("records")]

    antes = len(dados._ler_supabase())
    tabela = dados._cliente().table(dados.TABELA)
    for i in range(0, len(linhas), 200):
        tabela.upsert(linhas[i:i + 200], on_conflict="id_pessoa", ignore_duplicates=True,
                      returning=ReturnMethod.minimal).execute()
    depois = len(dados._ler_supabase())

    print(f"Excel: {len(linhas)} linhas | banco antes: {antes} | banco agora: {depois} "
          f"| novas: {depois - antes}")
    if depois < len(linhas):
        sys.exit("Atenção: o banco tem menos linhas que o Excel.")


if __name__ == "__main__":
    main()
