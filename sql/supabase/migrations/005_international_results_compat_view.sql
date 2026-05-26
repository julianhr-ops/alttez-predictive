create or replace view public.international_results_compat as
select
    m.match_date as date,
    ht.name as home_team,
    at.name as away_team,
    m.home_score,
    m.away_score,
    tr.name as tournament,
    m.city,
    m.venue_country_name as country,
    m.is_neutral as neutral
from public.matches as m
join public.teams as ht
    on ht.id = m.home_team_id
join public.teams as at
    on at.id = m.away_team_id
left join public.tournaments as tr
    on tr.id = m.tournament_id
where m.status = 'finished';
