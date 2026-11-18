alter table public.nba_predictions drop constraint if exists pk_nba_predictions;
alter table public.nba_predictions
    add constraint pk_nba_predictions primary key (game_id, model_version, served_model);
