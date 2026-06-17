-- Durable model artifact store.
-- The model binary (.joblib) used to live only on the ephemeral training host and was
-- git-ignored, so the active XGBoost artifact was never present at brief-generation time
-- and inference silently fell back to Poisson. Persisting the bytes here, keyed by the
-- same version_key as public.model_versions, lets any environment hydrate the active model.

create table if not exists public.model_artifacts (
    version_key    text primary key,
    artifact_name  text not null,
    artifact_bytes bytea not null,
    byte_size      bigint not null,
    sha256         text,
    created_at     timestamptz not null default now()
);

create index if not exists idx_model_artifacts_created_at
    on public.model_artifacts(created_at desc);
