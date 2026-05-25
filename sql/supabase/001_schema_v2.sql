create extension if not exists pgcrypto;

create table if not exists public.teams (
    id uuid primary key default gen_random_uuid(),
    name varchar(120) not null,
    normalized_name varchar(120) not null,
    fifa_code varchar(3),
    country_code varchar(3),
    is_national_team boolean not null default true,
    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now()),
    constraint uq_teams_normalized_name unique (normalized_name)
);

create index if not exists idx_teams_name on public.teams(name);

create table if not exists public.tournaments (
    id uuid primary key default gen_random_uuid(),
    name varchar(160) not null,
    normalized_name varchar(160) not null,
    competition_type varchar(40),
    region varchar(60),
    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now()),
    constraint uq_tournaments_normalized_name unique (normalized_name)
);

create index if not exists idx_tournaments_name on public.tournaments(name);

create table if not exists public.raw_matches_api (
    id uuid primary key default gen_random_uuid(),
    provider varchar(50) not null,
    external_match_id varchar(100) not null,
    fetched_at timestamptz not null default timezone('utc', now()),
    payload jsonb not null,
    payload_hash varchar(128),
    processed boolean not null default false,
    processed_at timestamptz,
    created_at timestamptz not null default timezone('utc', now()),
    constraint uq_raw_matches_provider_match unique (provider, external_match_id)
);

create index if not exists idx_raw_matches_processed on public.raw_matches_api(processed, fetched_at);
create index if not exists idx_raw_matches_payload_gin on public.raw_matches_api using gin(payload);

create table if not exists public.matches (
    id uuid primary key default gen_random_uuid(),
    provider varchar(50),
    external_match_id varchar(100),
    historical_fingerprint varchar(64),
    match_date date not null,
    kickoff_at timestamptz,
    season_label varchar(30),
    stage varchar(60),
    status varchar(30) not null default 'finished',
    tournament_id uuid references public.tournaments(id) on delete restrict,
    home_team_id uuid not null references public.teams(id) on delete restrict,
    away_team_id uuid not null references public.teams(id) on delete restrict,
    venue_name varchar(160),
    venue_country_code varchar(3),
    venue_country_name varchar(120),
    city varchar(120),
    referee_name varchar(120),
    home_score integer not null check (home_score >= 0),
    away_score integer not null check (away_score >= 0),
    is_neutral boolean not null default false,
    source_quality varchar(20) not null default 'historical_csv',
    ingested_at timestamptz not null default timezone('utc', now()),
    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now()),
    constraint check_different_teams check (home_team_id <> away_team_id),
    constraint check_status check (status in ('scheduled', 'in_progress', 'finished', 'postponed', 'cancelled')),
    constraint check_source_quality check (
        source_quality in ('historical_csv', 'api_raw', 'api_enriched', 'manual')
    ),
    constraint uq_matches_provider_match unique (provider, external_match_id),
    constraint uq_matches_historical_fingerprint unique (historical_fingerprint)
);

create index if not exists idx_matches_match_date on public.matches(match_date desc);
create index if not exists idx_matches_kickoff_at on public.matches(kickoff_at desc);
create index if not exists idx_matches_tournament on public.matches(tournament_id, match_date desc);
create index if not exists idx_matches_home_team on public.matches(home_team_id, match_date desc);
create index if not exists idx_matches_away_team on public.matches(away_team_id, match_date desc);
create index if not exists idx_matches_status on public.matches(status);

create table if not exists public.match_team_stats (
    id uuid primary key default gen_random_uuid(),
    match_id uuid not null references public.matches(id) on delete cascade,
    team_id uuid not null references public.teams(id) on delete restrict,
    opponent_team_id uuid not null references public.teams(id) on delete restrict,
    is_home boolean not null,
    formation varchar(10),
    xg numeric(6,2),
    possession numeric(5,2) check (possession >= 0 and possession <= 100),
    shots_total integer check (shots_total is null or shots_total >= 0),
    shots_on_target integer check (shots_on_target is null or shots_on_target >= 0),
    corners integer check (corners is null or corners >= 0),
    fouls integer check (fouls is null or fouls >= 0),
    yellow_cards integer check (yellow_cards is null or yellow_cards >= 0),
    red_cards integer check (red_cards is null or red_cards >= 0),
    goals_scored integer not null check (goals_scored >= 0),
    goals_conceded integer not null check (goals_conceded >= 0),
    source_quality varchar(20) not null default 'historical_csv',
    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now()),
    constraint uq_match_team_stats unique (match_id, team_id),
    constraint check_stats_different_teams check (team_id <> opponent_team_id),
    constraint check_stats_source_quality check (
        source_quality in ('historical_csv', 'api_raw', 'api_enriched', 'manual')
    )
);

create index if not exists idx_match_team_stats_match on public.match_team_stats(match_id);
create index if not exists idx_match_team_stats_team on public.match_team_stats(team_id, is_home);
create index if not exists idx_match_team_stats_opponent on public.match_team_stats(opponent_team_id);

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = timezone('utc', now());
    return new;
end;
$$;

drop trigger if exists trg_teams_updated_at on public.teams;
create trigger trg_teams_updated_at
before update on public.teams
for each row execute function public.set_updated_at();

drop trigger if exists trg_tournaments_updated_at on public.tournaments;
create trigger trg_tournaments_updated_at
before update on public.tournaments
for each row execute function public.set_updated_at();

drop trigger if exists trg_matches_updated_at on public.matches;
create trigger trg_matches_updated_at
before update on public.matches
for each row execute function public.set_updated_at();

drop trigger if exists trg_match_team_stats_updated_at on public.match_team_stats;
create trigger trg_match_team_stats_updated_at
before update on public.match_team_stats
for each row execute function public.set_updated_at();
