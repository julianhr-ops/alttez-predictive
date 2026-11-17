# Datos y modelos

## Clases de datos

| Clase | Ubicacion o sistema | Uso |
| --- | --- | --- |
| Historico internacional | Supabase o fallback local | Entrenamiento y perfiles |
| Agenda/resultados live | Stack de proveedores | Prediccion diaria y cierre |
| Configuracion Mundial 2026 | `storage/world_cup_2026/` | Simulacion |
| NBA: calendario y lineas de cierre | hoopR-nba-data (CC BY 4.0), cache en `artifacts/nba/` | Entrenamiento y validacion NBA |
| NBA: partidos proximos y linea | ESPN scoreboard, una fecha por llamada | Prediccion diaria NBA |
| Artefactos de modelos | Local/Supabase segun servicio | Inferencia versionada |
| Reportes y auditorias | `artifacts/` | Salidas regenerables |

No inferir que un CSV de la raiz sea automaticamente canonico. Seguir la ruta de
carga del modulo y sus variables de entorno.

## Normalizacion y tiempo

- Mantener nombres canonicos de equipos al cruzar proveedores.
- Tratar fechas y kickoff como datos temporales explicitos.
- Deduplicar por identidad de fixture, no por el indice de una API.
- Conservar procedencia y estado del proveedor cuando el contrato lo permita.
- No entrenar con informacion posterior al instante de prediccion.

## Contrato de prediccion

Los consumidores esperan probabilidades 1X2, intensidades de gol, marcador
principal, identificacion de modelo y diagnosticos de soporte/fallback. Las
probabilidades deben ser finitas, no negativas y sumar aproximadamente uno. Un
cambio de payload requiere revisar brief, dashboard, memoria y pruebas.

## Capas de modelo

1. Poisson sirve como baseline y fallback operativo.
2. Dixon-Coles, bivariado y jerarquico refinan distribuciones segun artefactos.
3. XGBoost es una familia gobernada por el ciclo challenger.
4. El comite combina miembros disponibles con pesos auditables.
5. La calibracion post-hoc ajusta probabilidades si existe artefacto valido.
6. La optimizacion usa reglas de puntos y no siempre el marcador mas probable.

No comparar modelos solo por accuracy. El proyecto usa Brier, Log Loss, RPS,
calibracion y metricas operativas; la politica concreta vive en
`training/challenger_cycle.py`. Todo modelo se compara con baselines (uniforme,
tasa base, Elo) sobre los mismos partidos, con bootstrap pareado
(`alttez/evaluation/`). La validacion walk-forward mostro que el campeon Poisson
no supera a Elo solo y que una regla Elo ordinal, elegida con 2018 a 2022 y
confirmada en 2023 a 2025, si lo supera (`docs/validation_soccer.md`). Esa regla
se sirve por defecto (`inference/serving_policy.py`, `ALTTEZ_SERVING_POLICY=off`
la desactiva). Elo NBA no supera la linea de cierre (`docs/nba.md`).

## Cold start y fallbacks

La ausencia de datos, base o artefactos es un estado esperado. Los componentes
deben degradar de forma visible y testeable:

- proveedor siguiente en la cadena;
- historico local solo si esta permitido;
- predictor Poisson si el activo no puede construirse;
- calibracion neutra si no hay artefacto;
- diagnostico de bajo soporte cuando faltan partidos.

No capturar excepciones sin conservar la razon funcional del fallback.

## Cambiar features

1. Modificar `MODEL_FEATURE_COLUMNS` y el calculo en `features/klement.py`.
2. Revisar la generacion del dataset de entrenamiento.
3. Revisar cada predictor que consume el vector.
4. Definir compatibilidad o nueva version de esquema del artefacto.
5. Probar orden, columnas ausentes, valores no finitos y paridad.
6. Documentar cualquier variable nueva en `.env.example`.

## Cambiar registro o SQL

Revisar conjuntamente la migracion, `prediction_memory.py`, activacion y rollback,
monitoreo, compatibilidad con versiones antiguas y fixtures de migracion.
