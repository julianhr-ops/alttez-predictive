# Arquitectura

## Mapa de directorios

| Ruta | Responsabilidad |
| --- | --- |
| `alttez/api/` | Carga de entorno, conexion SQLAlchemy y migraciones |
| `alttez/ingestion/` | Proveedores, contratos, agenda, historico y certificacion |
| `alttez/features/` | Features, ELO, Poisson, jerarquicos y preprocesamiento |
| `alttez/inference/` | Router, comite, calibracion y simulacion Mundial 2026 |
| `alttez/training/` | Entrenamiento, tuning y ciclo champion/challenger |
| `alttez/evaluation/` | Metricas y reglas de puntuacion |
| `alttez/operations/` | Memoria, registro, artefactos, expansion y alertas |
| `alttez/reports/` | Brief diario, Mundial 2026 y entregas |
| `alttez/dashboard/` | Aplicacion Streamlit y monitoreo |
| `scripts/` | Entradas operativas, no logica de dominio reutilizable |
| `sql/supabase/migrations/` | Historial SQL numerado y solo hacia adelante |
| `tests/` | Pruebas unitarias y de contratos |
| `storage/` | Insumos locales versionados y recursos estaticos |
| `artifacts/` | Salidas generadas en ejecucion |

## Ingestion

`provider_stack.py` construye una cadena ordenada de proveedores. Cada proveedor
implementa el contrato de `providers/base.py` y decide si esta configurado antes de
consultarse. La agenda diaria usa su propia variable de stack. Los contratos
normalizados estan en `ingestion/contracts.py`; la deduplicacion y lectura por fecha
estan en `live_matches.py`.

El historico se resuelve en `international.py`: puede consultar Supabase y, segun
configuracion, recurrir a datos locales. La certificacion convierte respuestas en
payloads auditables bajo `artifacts/certification/`.

Al agregar un proveedor:

1. Implementar el contrato base y `is_configured`.
2. Registrar el proveedor en el stack.
3. Normalizar equipos, fechas y estado antes de cruzar fuentes.
4. Agregar pruebas de contrato, fallos y deduplicacion.
5. Actualizar `.env.example` y `docs/ai/configuration.md`.

## Features e inferencia

`features/klement.py::MODEL_FEATURE_COLUMNS` define el esquema compartido por
entrenamiento e inferencia. No se permiten copias independientes de esa lista.

`inference/router.py::build_active_predictor` consulta el registro y construye el
predictor disponible. Debe recurrir a Poisson cuando la base, el registro o un
artefacto no estan disponibles, exponiendo `used_fallback` y `fallback_reason`.

`prediction_committee.py` combina champion, challenger y baseline.
`committee_weighting.py` decide pesos usando metricas recientes y puede deferir al
champion. `calibration.py` aplica calibracion post-hoc cuando corresponde. Los
marcadores se apoyan en `experiments/tournament_oracle.py` y
`evaluation/regulation_points.py`.

## Entrenamiento y gobierno

`training/challenger_cycle.py` arma datos, entrena el retador, evalua contra el
incumbente y produce una decision de gobierno. La promocion automatica permanece
desactivada salvo configuracion explicita. Las versiones se registran mediante
`operations/prediction_memory.py`; los binarios se manejan mediante
`operations/artifact_store.py`.

No activar un modelo solo porque un entrenamiento termino. Elegibilidad, familia
soportada, metricas, artefacto y politica de promocion son controles separados.

## Persistencia

`api/db.py` resuelve la URL de base. `api/migrations.py` descubre y aplica SQL. Las
migraciones actuales van de `001` a `012`; verificar siempre el directorio porque
la cadena seguira creciendo.

`operations/prediction_memory.py` guarda corridas y predicciones, cierra
evaluaciones, calcula monitoreo y administra versiones activas.

## Presentacion y entrega

El brief diario se organiza en `reports/daily_brief/`: payload, confiabilidad,
formato, HTML, entrega y escritura. El gate puede impedir el envio de una corrida
no confiable, pero no debe ocultar silenciosamente partidos. Streamlit debe
consumir servicios de dominio; la logica estadistica no debe vivir solo en la UI.
