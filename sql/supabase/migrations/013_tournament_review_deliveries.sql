create table if not exists public.tournament_review_deliveries (
    tournament_key varchar(160) primary key,
    tournament_name varchar(200) not null,
    date_from date not null,
    date_to date not null,
    delivery_status varchar(20) not null default 'pending',
    report_generated_at timestamptz,
    delivered_at timestamptz,
    evaluated_predictions integer not null default 0,
    rating numeric(6,2),
    report_payload jsonb not null default '{}'::jsonb,
    error_message text,
    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now()),
    constraint check_tournament_review_delivery_status check (
        delivery_status in ('pending', 'delivered', 'failed', 'skipped')
    )
);

create index if not exists idx_tournament_review_deliveries_status
    on public.tournament_review_deliveries(delivery_status, date_to desc);

drop trigger if exists trg_tournament_review_deliveries_updated_at
    on public.tournament_review_deliveries;
create trigger trg_tournament_review_deliveries_updated_at
before update on public.tournament_review_deliveries
for each row execute function public.set_updated_at();
