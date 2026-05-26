alter table public.matches add column if not exists group_label varchar(60);
alter table public.matches add column if not exists competition_group_id uuid references public.competition_groups(id) on delete set null;
alter table public.matches add column if not exists competition_class varchar(40);
alter table public.matches add column if not exists competition_normalized_name varchar(160);

do $$
begin
    if not exists (
        select 1
        from pg_constraint
        where conname = 'check_competition_class'
          and conrelid = 'public.matches'::regclass
    ) then
        alter table public.matches
        add constraint check_competition_class check (
            competition_class is null
            or competition_class in ('friendly', 'world_cup', 'qualifiers', 'continental', 'other_international')
        );
    end if;
end;
$$;
