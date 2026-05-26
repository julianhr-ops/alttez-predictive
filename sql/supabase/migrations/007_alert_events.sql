create table if not exists public.alert_events (
    id uuid primary key default gen_random_uuid(),
    alert_key varchar(180) not null,
    batch_id varchar(100) not null,
    target_date date not null,
    provider_used varchar(50),
    alert_scope varchar(40) not null,
    alert_level varchar(20) not null,
    alert_code varchar(60) not null,
    title text not null,
    message text not null,
    storage_status varchar(20) not null default 'pending',
    delivery_channel varchar(40) not null default 'dashboard',
    delivery_status varchar(20) not null default 'skipped',
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now()),
    constraint uq_alert_events_alert_key unique (alert_key),
    constraint check_alert_events_level check (alert_level in ('info', 'warning', 'critical')),
    constraint check_alert_events_storage_status check (
        storage_status in ('pending', 'stored', 'skipped', 'failed')
    ),
    constraint check_alert_events_delivery_status check (
        delivery_status in ('pending', 'delivered', 'failed', 'skipped')
    )
);

create index if not exists idx_alert_events_target_date on public.alert_events(target_date desc);
create index if not exists idx_alert_events_level on public.alert_events(alert_level);
create index if not exists idx_alert_events_status on public.alert_events(storage_status, delivery_status);
create index if not exists idx_alert_events_batch on public.alert_events(batch_id);

drop trigger if exists trg_alert_events_updated_at on public.alert_events;
create trigger trg_alert_events_updated_at
before update on public.alert_events
for each row execute function public.set_updated_at();
