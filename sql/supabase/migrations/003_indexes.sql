create index if not exists idx_matches_match_date on public.matches(match_date desc);
create index if not exists idx_matches_kickoff_at on public.matches(kickoff_at desc);
create index if not exists idx_matches_tournament on public.matches(tournament_id, match_date desc);
create index if not exists idx_matches_home_team on public.matches(home_team_id, match_date desc);
create index if not exists idx_matches_away_team on public.matches(away_team_id, match_date desc);
create index if not exists idx_matches_status on public.matches(status);
create index if not exists idx_matches_competition_class on public.matches(competition_class, match_date desc);
create index if not exists idx_matches_group_label on public.matches(group_label);
