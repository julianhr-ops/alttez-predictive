-- 014_prediction_raw_probabilities.sql
-- Guarda las probabilidades 1X2 CRUDAS (previas a la calibracion) junto a las calibradas.
-- La temperatura se ajusta sobre las probabilidades crudas; ajustarla sobre las ya
-- calibradas compone calibraciones. Nullable: las filas anteriores a esta migracion no
-- las tienen y el ajuste cae a las almacenadas (ya calibradas) solo para esas filas.

alter table public.match_predictions
    add column if not exists raw_predicted_home_win_prob numeric(10,6),
    add column if not exists raw_predicted_draw_prob numeric(10,6),
    add column if not exists raw_predicted_away_win_prob numeric(10,6);
