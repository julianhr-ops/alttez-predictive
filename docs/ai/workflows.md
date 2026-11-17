# Flujos de desarrollo y operacion

## Preparacion local

```bash
python -m venv venv
python -m pip install -r requirements.txt
cp .env.example .env
```

Activar antes el entorno (`source venv/bin/activate` en Linux y macOS,
`venv\Scripts\Activate.ps1` en Windows). Se requiere Python 3.13 o superior.

Completar solo las variables necesarias. `alttez.api.settings.load_local_env`
carga `.env`. Nunca versionar ese archivo.

## Ciclo de desarrollo

```bash
python -m unittest discover -s tests
python -m unittest tests.test_prediction_committee
python -m ruff check .
python scripts/apply_supabase_migrations.py
```

Las pruebas usan `unittest`. Ruff no esta en `requirements.txt`: instalar `pip install ruff==0.17.5` (la version
de CI). CI lo ejecuta como job bloqueante.
Las pruebas que requieren red o una base real no deben incorporarse por accidente
a la suite unitaria.

## Dashboard

```bash
python -m streamlit run alttez/dashboard/app.py
```

Separar calculos y transformaciones de Streamlit para probarlos sin navegador.

## Proveedores

```bash
python -m alttez.ingestion.provider_healthcheck --json
python scripts/smoke_live_pipeline.py --date AAAA-MM-DD
```

El smoke sin `--write` es la opcion inicial. `--write` persiste datos.

Estados de salida del smoke: `0` exito con agenda, `3` consulta valida sin agenda
(`idle/no_schedule`) y `1` fallo operativo. El codigo `3` no representa un fallo
del modelo ni debe disparar una alerta de incidente.

## Migraciones

```bash
# Validar, sin aplicar
python scripts/apply_supabase_migrations.py

# Aplicar con SUPABASE_DB_URL configurada
python scripts/apply_supabase_migrations.py --apply
```

Crear el siguiente archivo numerico y agregar pruebas en
`tests/test_migrations_runner.py`. No modificar archivos ya aplicables en ambientes
compartidos.

## Proceso diario

```powershell
.\scripts\run_daily_process.ps1 -ApplyMigrations -WriteSmoke -Date AAAA-MM-DD
```

GitHub lo ejecuta a las 02:00 UTC y permite despacho manual. Puede consultar red,
escribir en Supabase, entrenar, generar artefactos, entregar briefs y alertar. Leer
el PowerShell y `.github/workflows/daily_process.yml` antes de cambiar su orden.

Cuando no hay agenda internacional, el proceso conserva el backfill de resultados y
el cierre de torneos, y luego termina correctamente como `idle/no_schedule`. En ese
caso no ejecuta calibracion, brief diario ni ciclo challenger, y no envia una
alerta critica. La etapa NBA (`scripts/nba_daily.py`, no fatal) corre tambien en
ese camino.

## Modelos

```bash
python scripts/run_challenger_cycle.py --date AAAA-MM-DD
python scripts/fit_probability_calibration.py --help
python scripts/evaluate_calibration.py --help
python scripts/activate_model_version.py --help
python scripts/rollback_active_model.py --help
```

## Validacion y NBA

```bash
python scripts/validate_soccer.py --help
python scripts/nba_validate.py
python scripts/nba_daily.py --dry-run
```

`validate_soccer.py` tarda decenas de minutos con `--refit-days 1`. `nba_validate.py`
y `nba_daily.py` descargan datos gratuitos; con `--dry-run` no escriben en la base.
Resultados y politica de modelo servido: `docs/validation_soccer.md` y `docs/nba.md`.

Activacion, rollback, escritura, entrega y migraciones tienen efectos externos. No
ejecutarlos como simple verificacion.

## Criterio de finalizacion

- Cambio pequeno: prueba enfocada y lint de archivos afectados.
- Cambio compartido: suite completa y lint.
- Cambio SQL: validacion de migraciones y pruebas del runner.
- Cambio de provider: contratos, error/fallback y deduplicacion.
- Cambio de modelo: paridad train/inference, fallback, metricas y artefactos.
- Cambio de UI/reporte: pruebas de helpers y verificacion manual.
- Cambio operativo: probar modo sin escritura antes de efectos externos.
