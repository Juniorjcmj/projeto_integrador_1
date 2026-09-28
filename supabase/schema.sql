-- Tabela de desligamentos a pedido (entrevistas de saída) — Painel de Evasão de Militares.
-- Rode uma vez no SQL Editor do Supabase. É idempotente: pode rodar de novo sem erro.
--
-- Modelo de acesso: o painel (Streamlit) roda no servidor e usa a chave secreta (service_role).
-- RLS fica ativo SEM políticas públicas, então anon/authenticated não leem nem gravam nada;
-- os dados de militares nunca ficam acessíveis pela chave publicável.

create table if not exists public.desligamentos (
  id                  bigint generated always as identity primary key,
  id_pessoa           text not null unique,               -- PESSOA_001 (real) ou SINT_001 (sintético)
  ano                 smallint check (ano between 1990 and 2100),
  mes                 smallint check (mes between 1 and 12),
  data_desligamento   date,                                -- nula quando só se sabe mês/ano
  posto_quadro        text,
  motivo_principal    text not null default 'Não Informado'
                      check (motivo_principal in ('Concurso Público', 'Família / Vida Pessoal', 'Financeiro',
                                                  'Saúde', 'Outro', 'Não Informado')),
  menciona_concurso_publico           boolean not null default false,
  menciona_saude                      boolean not null default false,
  menciona_familia_vida_pessoal       boolean not null default false,
  menciona_financeiro                 boolean not null default false,
  menciona_estudos_pos_graduacao      boolean not null default false,
  menciona_insatisfacao_profissional  boolean not null default false,
  origem_registro     text not null default 'app' check (origem_registro in ('app', 'importacao')),
  criado_em           timestamptz not null default now()
);

comment on table public.desligamentos is
  'Desligamentos a pedido de militares. Registros SINT_* são sintéticos (Gaussian Copula).';

alter table public.desligamentos enable row level security;

-- Acesso explícito e mínimo. O service_role ignora RLS, então os privilégios são a única barreira:
-- revogamos tudo (inclusive o que os privilégios padrão do projeto concedem) e devolvemos só
-- leitura e inserção — o painel não altera nem apaga registros. Correções manuais continuam
-- possíveis pelo Table Editor do dashboard, que usa o papel postgres.
revoke all on public.desligamentos from anon, authenticated, service_role;
grant select, insert on public.desligamentos to service_role;

-- Conferência: deve mostrar rls = true e service_role só com INSERT e SELECT.
select c.relrowsecurity as rls,
       (select string_agg(g.grantee || ':' || g.privilege_type, ', ' order by g.grantee, g.privilege_type)
          from information_schema.role_table_grants g
         where g.table_schema = 'public' and g.table_name = 'desligamentos'
           and g.grantee in ('anon', 'authenticated', 'service_role')) as privilegios
  from pg_class c
 where c.oid = 'public.desligamentos'::regclass;
