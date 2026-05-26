create table if not exists public.certification_batches (
    id uuid primary key default gen_random_uuid(),
    batch_id varchar(100) not null,
    target_date date not null,
    provider_used varchar(50),
    certification_status varchar(20) not null,
    certified boolean not null default false,
    total_records integer not null default 0 check (total_records >= 0),
    certified_records integer not null default 0 check (certified_records >= 0),
    rejected_records integer not null default 0 check (rejected_records >= 0),
    coverage_pct numeric(5,2) not null default 0 check (coverage_pct >= 0 and coverage_pct <= 100),
    quarantine_reason text,
    report_path text,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now()),
    constraint uq_certification_batches_batch_id unique (batch_id),
    constraint check_certification_batches_status check (
        certification_status in ('certified', 'quarantined', 'skipped', 'error')
    )
);

create index if not exists idx_certification_batches_target_date on public.certification_batches(target_date desc);
create index if not exists idx_certification_batches_status on public.certification_batches(certification_status);
create index if not exists idx_certification_batches_provider on public.certification_batches(provider_used);

drop trigger if exists trg_certification_batches_updated_at on public.certification_batches;
create trigger trg_certification_batches_updated_at
before update on public.certification_batches
for each row execute function public.set_updated_at();
