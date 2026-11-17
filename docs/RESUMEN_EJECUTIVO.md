# Resumen ejecutivo — ALTTEZ Predictive Engine

> Documento de estado, riesgos y roadmap. Rama activa: `desarrollo`. Última revisión: 2026-06-16.

## 1. Qué es ALTTEZ

Motor de predicción de fútbol (reescritura del antiguo ProphitBet) enfocado en **selecciones y torneos**. Flujo de punta a punta:

```
ingesta multi-proveedor → Supabase (Postgres) → features ELO/Poisson/Klement
   → router de inferencia (1X2 + marcador exacto, optimizado contra reglas "polla")
   → brief diario (Telegram) + dashboard Streamlit
```

Un ciclo de **entrenamiento/gobernanza** separado decide qué modelo sirve el router.

## 2. Estado funcional actual

| Área | Estado | Detalle |
|------|--------|---------|
| **Ingesta** | Operativa | Stack de 4 providers (`api_football`, `football_data_org`, `statsbomb_open`, `local_json`), todos con `is_configured()`/`fetch_matches` reales y fall-through ordenado. Escriben a `raw_matches_api`. |
| **Inferencia** | Operativa | `router.build_active_predictor` resuelve el modelo activo desde `model_versions`; **fallback garantizado a Poisson** si el artefacto no carga (deja traza `used_fallback`/`fallback_reason`). Comité de predicción (`prediction_committee.py`) cableado en router, payload y render. |
| **Gobernanza** | Operativa | `challenger_cycle.py`: champion reconstruido con la configuración de producción (si no carga, el ciclo se rechaza); el Brier por partido decide con bootstrap pareado sobre los mismos partidos (IC completo bajo 0 y muestra mínima), Log Loss y RPS son guardrails, y el reporte incluye baselines uniforme/tasa base/Elo. Auto-promoción **OFF** salvo `ALTTEZ_AUTO_PROMOTE_MODELS=true`. Estados `eligible`/`shadow`/`active`. |
| **Persistencia** | Operativa | Cadena de migraciones forward-only `001…009`. Tablas de memoria de predicción: `prediction_runs`, `match_predictions`, `prediction_evaluations`, `model_versions`. |
| **Operación** | Operativa | Proceso diario en `scripts/run_daily_process.ps1` / `.github/workflows/daily_process.yml` (02:00 UTC). Brief a Telegram + alertas de incidente (opt-in por env). Simulador Mundial 2026. |
| **Calidad** | Parcial | 21 archivos de test (unittest) sobre 60 módulos. CI corre tests + valida la cadena de migraciones. |
| **Transform raw→canónico** | **En progreso** | La promoción de `raw_matches_api` a tablas canónicas `matches`/`match_team_stats` sigue pendiente (ver R2). |

## 3. Cosas a tener en cuenta (riesgos / deuda técnica)

Cada ítem tiene evidencia concreta en el repo.

| # | Tema | Evidencia | Riesgo |
|---|------|-----------|--------|
| **R1** | README.md duplicado | `# ALTTEZ Predictive Engine` aparece 2× — todo el contenido se repite (~línea 388 en adelante) | Confusión y drift de documentación |
| **R2** | Transform raw→canónico incompleto | `grep "canonical"` solo arroja `canonical_team_name` (normalización de nombres), no un transformador a la tabla `matches`; el README lo marca "in progress" | Los providers llenan `raw_matches_api`, pero la promoción a tablas canónicas es manual/pendiente |
| **R3** | `predict_matches.py` rastreado en raíz | Está en git; importa `build_prediction_committee`; lee CSVs **gitignoreados** (`poisson_team_metrics.csv`, `mundial_2026_fixtures.csv`) | Script tipo scratch en raíz que depende de datos no versionados → no reproducible en un checkout limpio/CI |
| **R4** | Capturas amplias `except Exception` | 46 ocurrencias en 15 archivos (`prediction_memory` 11, `live_matches` 8) | Algunas son fallback intencional (router); otras pueden ocultar errores reales sin loggear |
| **R5** | Gaps de cobertura de tests | Sin test directo: implementaciones de los 4 providers, `dashboard/app.py`, `scraping/`, script de simulación Mundial | Regresiones silenciosas en ingesta y UI |
| **R6** | CI sin lint ni type-check | `ci.yml` solo corre `unittest` + validación de migraciones | Estilo y typing no garantizados; bugs de tipo se escapan |
| **R7** | Dependencias pesadas posiblemente sin uso en la ruta diaria | `selenium`, `shap`, `boruta`, `optuna`, `seaborn` en `requirements.txt` (UTF-16, ~90 pins) | Mayor superficie de instalación, build y seguridad de la necesaria |
| **R8** | Fechas/hosts hardcoded de 2026 | `WORLD_CUP_2026_HOST_TEAMS`, fechas `2026-xx` en CLIs y ejemplos | Acoplamiento a un torneo concreto; difícil reusar el motor para otro |
| **R9** | Archivos monolíticos | `dashboard/app.py` 2026 líneas / 43 funciones; `ingestion/live_matches.py` 1488 líneas / 43 defs mezclando normalización, parseo por-provider, clasificación, filtros, upserts y certificación. Le siguen `prediction_memory.py` (856), `klement.py` (740), `challenger_cycle.py` (706) | Difícil de leer, testear y modificar; `live_matches.py` además es hot path. Sin tests (cruza con R5) |

**Bien resuelto (no requiere acción):** `.env` correctamente gitignoreado (no rastreado); `.gitignore` amplio y consistente; fallback de inferencia robusto; gobernanza con auto-promoción desactivada por defecto.

## 4. Posibles cambios — roadmap priorizado

### Rápidos (bajo esfuerzo, alto valor)
- **C1 (R1)** — ✅ Hecho. `README.md` de-duplicado (de 644 a 387 líneas, una sola copia).
- **C2 (R3)** — ✅ Hecho. `predict_matches.py` movido a `scripts/predict_matches.py` (`git mv`). Pendiente: aún lee CSVs gitignoreados desde la raíz (no reproducible en CI limpio) — resolver junto a C7.
- **C3 (R6)** — ✅ Hecho. Job `lint` con `ruff` añadido a `ci.yml` como **no bloqueante** (`continue-on-error: true`).

### Medios
- **C4 (R5)** — Tests para los 4 providers (`fetch_matches`/`is_configured` con fixtures JSON) y un smoke del script de simulación Mundial.
- **C5 (R4)** — Auditar `except Exception` en `prediction_memory`/`live_matches`: estrechar a excepciones esperadas y loggear el resto.
- **C6 (R7)** — Revisar dependencias; mover scraping/research (`selenium`, `boruta`, `shap`) a un `requirements-dev.txt` opcional si no están en la ruta diaria.

### Estructurales (alto esfuerzo)
- **C7 (R2)** — Completar la capa de transformación `raw_matches_api` → tablas canónicas `matches`/`match_team_stats` (cierra el gap "in progress") y agendar la ingesta diaria.
- **C8 (R8)** — Parametrizar torneo/fechas vía configuración en lugar de constantes 2026, para reusar el motor en otros torneos.
- **C9 (R9)** — Romper monolitos: dividir `live_matches.py` por responsabilidad (parsers por-provider, normalización, persistencia/upserts, certificación en módulos separados) y extraer las vistas de `dashboard/app.py` a submódulos. Apoyarse en C4 (tests) antes de refactorizar el hot path.

## 5. Recomendación de arranque

1. **PR pequeño y seguro**: C1 → C2 → C3 (cosmético/infra, sin riesgo de regresión).
2. **Segundo PR**: C4 y C5 (endurecimiento de calidad).
3. **Hito propio**: C7 (transform canónico), con C6 y C8 acompañando según prioridad de producto.

Verificación de cualquier cambio de código: `venv\Scripts\python.exe -m unittest discover -s tests`.
