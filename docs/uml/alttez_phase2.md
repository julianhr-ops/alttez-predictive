# ALTTEZ Fase 2 — UML de arquitectura y flujo operativo

Este documento resume el estado actual del trabajo en ALTTEZ Fase 2:

- la rama activa es `desarrollo`
- la ingesta diaria ya corre con orquestación ligera en GitHub Actions
- la certificación de lotes escribe reporte local y ahora también persiste auditoría histórica en Supabase
- el dashboard muestra el estado de sincronización, cobertura y certificación
- la carpeta `screenshots/` quedó fuera del índice del repositorio y ya no forma parte del flujo del proyecto

La arquitectura se divide en cuatro capas principales:

1. **Extracción e ingesta**
2. **Validación, certificación y auditoría**
3. **Persistencia y consulta**
4. **Presentación y operación**

---

## 1) Diagrama de componentes

```mermaid
flowchart LR
    subgraph Providers["Fuentes externas"]
        A1[API-Football]
        A2[football-data.org]
        A3[StatsBomb Open Data]
        A4[Local JSON fallback]
    end

    subgraph Ingestion["Capa de ingesta"]
        B1[Provider Stack]
        B2[fetch_live_fixtures_from_stack]
        B3[parse_*_records]
        B4[upsert_raw_matches]
        B5[upsert_catalogs]
        B6[upsert_scheduled_matches]
    end

    subgraph Contracts["Contrato de datos"]
        C1[RawMatchPayload]
        C2[MatchTeamMetrics]
        C3[CertifiedMatch]
        C4[BatchCertification]
        C5[BatchRejectedMatch]
    end

    subgraph Certification["Certificación y auditoría"]
        D1[build_daily_certification_report]
        D2[write_certification_report]
        D3[build_certification_audit_row]
        D4[upsert_certification_audit]
        D5[record_operational_alerts]
    end

    subgraph Storage["Supabase / PostgreSQL"]
        E1[teams]
        E2[tournaments]
        E3[competition_groups]
        E4[raw_matches_api]
        E5[matches]
        E6[match_team_stats]
        E7[certification_batches]
        E8[alert_events]
    end

    subgraph Orchestration["Automatización diaria"]
        F1[GitHub Actions workflow]
        F2[scripts/run_daily_process.ps1]
        F3[scripts/smoke_live_pipeline.py]
        F4[scripts/shadow_readiness_check.py]
    end

    subgraph Presentation["Presentación"]
        G1[Streamlit Dashboard]
        G2[analysis_helpers]
        G3[Poisson / ELO / Oracle]
    end

    subgraph Observability["Trazabilidad"]
        H1[artifacts/certification/*.json]
        H2[README / docs / logs]
    end

    A1 --> B1
    A2 --> B1
    A3 --> B1
    A4 --> B1

    B1 --> B2 --> B3
    B3 --> B4
    B3 --> B5
    B3 --> B6

    B3 --> C1
    C1 --> C2
    C1 --> C3
    C1 --> C4
    C1 --> C5

    C4 --> D1 --> D2 --> H1
    C4 --> D3 --> D4 --> E7
    C4 --> D5 --> E8

    B4 --> E4
    B5 --> E1
    B5 --> E2
    B6 --> E3
    B6 --> E5
    C3 --> E6

    F1 --> F2 --> F3
    F2 --> F4
    F3 --> D1
    F3 --> D4

    E5 --> G1
    E6 --> G1
    E7 --> G1
    E8 --> G1
    G1 --> G2 --> G3

    H1 --> H2
    E7 --> H2
```

---

## 2) Diagrama de clases

```mermaid
classDiagram
    class ScheduledMatch {
        +str provider
        +str external_match_id
        +date match_date
        +str kickoff_at
        +str season_label
        +str stage
        +str group_label
        +str tournament_name
        +str competition_normalized_name
        +str competition_class
        +str tournament_region
        +str home_team
        +str away_team
        +str venue_name
        +str city
        +str venue_country_name
        +str referee_name
        +bool is_neutral
        +str status
        +int home_score
        +int away_score
        +dict payload
    }

    class MatchTeamMetrics {
        +float xg
        +float possession
        +int shots_total
        +int shots_on_target
        +int shots_off_target
        +int shots_blocked
        +int shots_inside_box
        +int shots_outside_box
        +int corners
        +int fouls
        +int yellow_cards
        +int red_cards
        +int offsides
        +int saves
        +int passes_completed
        +float pass_accuracy
        +coverage_pct()
    }

    class RawMatchPayload {
        +str provider
        +str external_match_id
        +date match_date
        +datetime kickoff_at
        +str tournament_name
        +str home_team
        +str away_team
        +str status
        +int home_score
        +int away_score
        +MatchTeamMetrics home_metrics
        +MatchTeamMetrics away_metrics
        +dict payload
        +coverage_pct()
        +hard_required_complete()
        +available_metric_count()
    }

    class CertifiedMatch {
        +str provider
        +str external_match_id
        +date match_date
        +datetime kickoff_at
        +str tournament_name
        +str home_team
        +str away_team
        +str status
        +int home_score
        +int away_score
        +MatchTeamMetrics home_metrics
        +MatchTeamMetrics away_metrics
        +bool certified
        +str certification_status
        +datetime certified_at
        +float coverage_pct
    }

    class BatchRejectedMatch {
        +str external_match_id
        +str provider
        +str reason
        +float coverage_pct
    }

    class BatchCertification {
        +str batch_id
        +date target_date
        +str provider_used
        +int total_records
        +int certified_records
        +int rejected_records
        +float coverage_pct
        +bool certified
        +str certification_status
        +str quarantine_reason
        +list certified_matches
        +list rejected_matches
        +certify_batch()
        +is_quarantined()
    }

    class RawMatchRecord {
        +str provider
        +str external_match_id
        +dict payload
        +datetime fetched_at
    }

    class ProviderFetchRequest {
        +date date_from
        +date date_to
        +str competition
    }

    class ProviderStack {
        +fetch_matches()
        +healthcheck()
    }

    class CertificationAuditRow {
        +str batch_id
        +date target_date
        +str provider_used
        +str certification_status
        +bool certified
        +int total_records
        +int certified_records
        +int rejected_records
        +float coverage_pct
        +str quarantine_reason
        +str report_path
        +dict payload
    }

    ScheduledMatch --> RawMatchRecord
    RawMatchPayload o-- MatchTeamMetrics
    CertifiedMatch o-- MatchTeamMetrics
    BatchCertification o-- CertifiedMatch
    BatchCertification o-- BatchRejectedMatch
    BatchCertification --> RawMatchPayload
    ProviderStack --> ProviderFetchRequest
    BatchCertification --> CertificationAuditRow
```

---

## 3) Diagrama de secuencia — proceso diario

```mermaid
sequenceDiagram
    autonumber
    participant GH as GitHub Actions
    participant SH as run_daily_process.ps1
    participant SR as shadow_readiness_check.py
    participant IN as ingest_live_matches_for_date
    participant PS as Provider Stack
    participant CP as Contracts / Certification
    participant DB as Supabase / PostgreSQL
    participant AU as certification_batches
    participant RPT as artifacts/certification
    participant UI as Streamlit Dashboard

    GH->>SH: Ejecuta job a las 02:00 UTC
    SH->>SR: Verifica entorno, sintaxis, migraciones y proveedores
    SR-->>SH: OK / FAIL

    alt preflight OK
        SH->>IN: smoke_live_pipeline --write
        IN->>PS: fetch_live_fixtures_from_stack()
        PS-->>IN: scheduled_matches + raw_records + provider_used
        IN->>DB: upsert_catalogs()
        IN->>DB: upsert_raw_matches()
        IN->>DB: upsert_scheduled_matches()
        IN->>CP: build_raw_match_payload()
        CP-->>IN: RawMatchPayload[]
        IN->>CP: build_daily_certification_report()
        CP-->>IN: BatchCertification
        IN->>RPT: write_certification_report()
        IN->>AU: build_certification_audit_row()
        IN->>DB: upsert_certification_audit()
        IN-->>SH: sync_result
        SH-->>GH: éxito del proceso
    else preflight FAIL
        SH-->>GH: falla temprana
    end

    UI->>DB: lee matches, match_team_stats, certification_batches
    UI->>RPT: muestra ruta del reporte si existe
```

---

## 4) Diagrama de despliegue

```mermaid
flowchart TB
    subgraph Dev["Entorno de desarrollo local"]
        L1[PowerShell / CMD launchers]
        L2[Streamlit dashboard]
        L3[Python venv]
        L4[Local artifacts/]
        L5[Git hooks / smoke tests]
    end

    subgraph CI["GitHub Actions"]
        C1[daily_process.yml]
        C2[runner temporal]
        C3[preflight + migrations + smoke]
    end

    subgraph Data["Supabase / PostgreSQL"]
        D1[teams]
        D2[tournaments]
        D3[competition_groups]
        D4[raw_matches_api]
        D5[matches]
        D6[match_team_stats]
        D7[certification_batches]
    end

    subgraph Ext["Servicios externos"]
        E1[API-Football]
        E2[football-data.org]
        E3[StatsBomb Open Data]
    end

    L1 --> L2
    L2 --> L3
    L3 --> L4
    L3 --> L5

    C1 --> C2 --> C3
    C3 --> D1
    C3 --> D2
    C3 --> D3
    C3 --> D4
    C3 --> D5
    C3 --> D6
    C3 --> D7

    E1 --> C3
    E2 --> C3
    E3 --> C3
```

---

## 5) Lectura rápida del estado actual

### Ya está operativo

- ingesta multi-provider con fallback
- normalización de equipos, torneos y grupos
- contrato de datos con `RawMatchPayload` y `CertifiedMatch`
- certificación de lotes diarios
- reporte JSON de certificación en `artifacts/certification/`
- auditoría persistida en `certification_batches`
- alertas operativas persistidas en `alert_events`
- dashboard con estado de sincronización y certificación
- workflow diario en GitHub Actions
- smoke test y readiness check

### Todavía puede crecer

- alertas automáticas para lotes cuarentenados
- notificación opcional por webhook para alertas críticas
- vista materializada de lotes certificados para consumo analítico
- dbt tests para validación de cobertura y rangos
- panel histórico de auditoría en el dashboard
- mejor observabilidad de proveedores y errores

---

## 6) Archivo relacionado

- [README.md](../../README.md)
- [docs/provider-stack.md](../provider-stack.md)
- [sql/supabase/migrations/006_certification_batches.sql](../../sql/supabase/migrations/006_certification_batches.sql)
