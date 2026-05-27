create table if not exists public.model_versions (
    id uuid primary key default gen_random_uuid(),
    version_key varchar(160) not null,
    model_version varchar(120) not null,
    model_family varchar(60) not null,
    training_cutoff date not null,
    trained_at timestamptz not null default timezone('utc', now()),
    evaluation_window_start date,
    evaluation_window_end date,
    brier_score numeric(10,6),
    log_loss numeric(10,6),
    accuracy_1x2 numeric(10,6),
    exact_score_hit_rate numeric(10,6),
    status varchar(20) not null default 'challenger',
    promotion_reason text,
    artifact_path text,
    metrics_payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now()),
    constraint uq_model_versions_version_key unique (version_key),
    constraint check_model_versions_status check (
        status in ('champion', 'challenger', 'rejected', 'archived')
    )
);

create index if not exists idx_model_versions_status on public.model_versions(status, trained_at desc);
create index if not exists idx_model_versions_family on public.model_versions(model_family, trained_at desc);

drop trigger if exists trg_model_versions_updated_at on public.model_versions;
create trigger trg_model_versions_updated_at
before update on public.model_versions
for each row execute function public.set_updated_at();

create table if not exists public.prediction_runs (
    id uuid primary key default gen_random_uuid(),
    run_id varchar(160) not null,
    model_version varchar(120) not null,
    model_family varchar(60) not null,
    target_date date not null,
    generated_at timestamptz not null,
    training_cutoff date not null,
    status varchar(20) not null default 'completed',
    summary_payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now()),
    constraint uq_prediction_runs_run_id unique (run_id),
    constraint check_prediction_runs_status check (
        status in ('draft', 'completed', 'failed')
    )
);

create index if not exists idx_prediction_runs_target_date on public.prediction_runs(target_date desc, generated_at desc);
create index if not exists idx_prediction_runs_model on public.prediction_runs(model_version, generated_at desc);

drop trigger if exists trg_prediction_runs_updated_at on public.prediction_runs;
create trigger trg_prediction_runs_updated_at
before update on public.prediction_runs
for each row execute function public.set_updated_at();

create table if not exists public.match_predictions (
    id uuid primary key default gen_random_uuid(),
    prediction_run_id uuid not null references public.prediction_runs(id) on delete cascade,
    match_id uuid not null references public.matches(id) on delete cascade,
    model_version varchar(120) not null,
    model_family varchar(60) not null,
    prediction_created_at timestamptz not null,
    match_date date not null,
    kickoff_at timestamptz,
    home_team varchar(120) not null,
    away_team varchar(120) not null,
    tournament_name varchar(160),
    competition_class varchar(40),
    predicted_outcome varchar(1) not null,
    predicted_home_win_prob numeric(10,6) not null,
    predicted_draw_prob numeric(10,6) not null,
    predicted_away_win_prob numeric(10,6) not null,
    predicted_scoreline varchar(20),
    predicted_lambda_home numeric(10,6),
    predicted_lambda_away numeric(10,6),
    support_label varchar(40),
    avg_confidence numeric(10,6),
    decision_tier varchar(40),
    decision_label varchar(60),
    best_market varchar(160),
    best_market_probability numeric(10,6),
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now()),
    constraint uq_match_predictions_run_match unique (prediction_run_id, match_id),
    constraint check_match_predictions_outcome check (predicted_outcome in ('H', 'D', 'A'))
);

create index if not exists idx_match_predictions_match_date on public.match_predictions(match_date desc);
create index if not exists idx_match_predictions_model on public.match_predictions(model_version, prediction_created_at desc);
create index if not exists idx_match_predictions_competition on public.match_predictions(competition_class, match_date desc);

drop trigger if exists trg_match_predictions_updated_at on public.match_predictions;
create trigger trg_match_predictions_updated_at
before update on public.match_predictions
for each row execute function public.set_updated_at();

create table if not exists public.prediction_evaluations (
    id uuid primary key default gen_random_uuid(),
    prediction_id uuid not null references public.match_predictions(id) on delete cascade,
    match_id uuid not null references public.matches(id) on delete cascade,
    evaluation_status varchar(20) not null default 'evaluated',
    actual_home_score integer not null check (actual_home_score >= 0),
    actual_away_score integer not null check (actual_away_score >= 0),
    actual_outcome varchar(1) not null,
    hit_1x2 boolean not null,
    hit_exact_score boolean not null,
    brier_1x2 numeric(10,6) not null,
    log_loss_1x2 numeric(10,6) not null,
    score_residual_home integer not null,
    score_residual_away integer not null,
    score_abs_error integer not null,
    evaluated_at timestamptz not null default timezone('utc', now()),
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now()),
    constraint uq_prediction_evaluations_prediction unique (prediction_id),
    constraint check_prediction_evaluations_status check (
        evaluation_status in ('evaluated', 'skipped')
    ),
    constraint check_prediction_evaluations_actual_outcome check (
        actual_outcome in ('H', 'D', 'A')
    )
);

create index if not exists idx_prediction_evaluations_match on public.prediction_evaluations(match_id, evaluated_at desc);
create index if not exists idx_prediction_evaluations_brier on public.prediction_evaluations(brier_1x2);

drop trigger if exists trg_prediction_evaluations_updated_at on public.prediction_evaluations;
create trigger trg_prediction_evaluations_updated_at
before update on public.prediction_evaluations
for each row execute function public.set_updated_at();
