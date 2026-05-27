alter table public.teams
    add column if not exists gdp_per_capita numeric(12,2);

alter table public.teams
    add column if not exists confederation_code varchar(12);

alter table public.teams
    add column if not exists macro_source varchar(40) not null default 'manual_csv';

alter table public.teams
    add column if not exists macro_updated_at timestamptz;
