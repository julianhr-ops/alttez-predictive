alter table public.model_versions
    add column if not exists is_promotion_eligible boolean not null default false;

alter table public.model_versions
    add column if not exists production_status varchar(20) not null default 'shadow';

alter table public.model_versions
    add column if not exists activation_mode varchar(20) not null default 'manual';

alter table public.model_versions
    add column if not exists activated_at timestamptz;

create index if not exists idx_model_versions_production_status
    on public.model_versions(production_status, trained_at desc);

do $$
begin
    if not exists (
        select 1
        from pg_constraint
        where conname = 'check_model_versions_production_status'
          and conrelid = 'public.model_versions'::regclass
    ) then
        alter table public.model_versions
        add constraint check_model_versions_production_status check (
            production_status in ('shadow', 'eligible', 'active', 'archived', 'blocked')
        );
    end if;
end;
$$;

do $$
begin
    if not exists (
        select 1
        from pg_constraint
        where conname = 'check_model_versions_activation_mode'
          and conrelid = 'public.model_versions'::regclass
    ) then
        alter table public.model_versions
        add constraint check_model_versions_activation_mode check (
            activation_mode in ('manual', 'auto', 'blocked')
        );
    end if;
end;
$$;
