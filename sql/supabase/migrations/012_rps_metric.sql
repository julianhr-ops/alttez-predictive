-- 012_rps_metric.sql
-- Anade el Ranked Probability Score (1X2 ordinal) a las evaluaciones de prediccion.
-- Nullable: las filas historicas cerradas antes de esta migracion no lo tienen;
-- se rellena hacia adelante en close_finished_prediction_evaluations.

alter table public.prediction_evaluations
    add column if not exists rps_1x2 numeric(10,6);

create index if not exists idx_prediction_evaluations_rps
    on public.prediction_evaluations(rps_1x2);
