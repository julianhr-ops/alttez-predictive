-- 015_nba_predictions.sql
-- Predicciones de partidos NBA (datos gratuitos: hoopR + ESPN). Una fila por
-- (game_id, model_version): se inserta una sola vez mientras el partido esta
-- programado (ON CONFLICT DO NOTHING) y despues solo se rellenan las columnas de
-- resultado al calificar. Nunca se reescribe una prediccion ya emitida.
-- market_spread es el margen esperado del local segun el mercado: positivo = local favorito.

create table if not exists public.nba_predictions (
    game_id bigint not null,
    game_date date not null,
    tipoff_utc timestamptz not null,
    home_team_id smallint not null,
    away_team_id smallint not null,
    model_version varchar(60) not null,
    served_model varchar(20) not null,
    p_home_elo numeric(8,6) not null,
    p_home_market numeric(8,6),
    p_home_served numeric(8,6) not null,
    expected_margin numeric(7,3) not null,
    market_spread numeric(6,2),
    market_total numeric(6,2),
    market_source varchar(60),
    created_at timestamptz not null default timezone('utc', now()),
    status varchar(12) not null default 'scheduled',
    home_score smallint,
    away_score smallint,
    home_win boolean,
    brier numeric(8,6),
    graded_at timestamptz,
    constraint pk_nba_predictions primary key (game_id, model_version),
    constraint check_nba_predictions_served_model check (served_model in ('blend', 'elo_platt')),
    constraint check_nba_predictions_status check (status in ('scheduled', 'graded')),
    constraint check_nba_predictions_teams check (home_team_id <> away_team_id),
    constraint check_nba_predictions_probabilities check (
        p_home_elo between 0 and 1
        and p_home_served between 0 and 1
        and (p_home_market is null or p_home_market between 0 and 1)
    ),
    constraint check_nba_predictions_blend_needs_market check (
        served_model <> 'blend' or p_home_market is not null
    ),
    constraint check_nba_predictions_grading check (
        (status = 'scheduled' and home_score is null and away_score is null
            and home_win is null and brier is null and graded_at is null)
        or (status = 'graded' and home_score is not null and away_score is not null
            and home_win is not null and brier is not null and graded_at is not null)
    )
);

create index if not exists idx_nba_predictions_status_date
    on public.nba_predictions(status, game_date desc);
create index if not exists idx_nba_predictions_game_date
    on public.nba_predictions(game_date desc);
