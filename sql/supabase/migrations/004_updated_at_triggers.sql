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

drop trigger if exists trg_competition_groups_updated_at on public.competition_groups;
create trigger trg_competition_groups_updated_at
before update on public.competition_groups
for each row execute function public.set_updated_at();

drop trigger if exists trg_match_team_stats_updated_at on public.match_team_stats;
create trigger trg_match_team_stats_updated_at
before update on public.match_team_stats
for each row execute function public.set_updated_at();
