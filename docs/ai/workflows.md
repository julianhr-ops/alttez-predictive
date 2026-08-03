# Flujos de desarrollo y operacion

## Preparacion local

```powershell
venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Completar solo las variables necesarias. `alttez.api.settings.load_local_env`
carga `.env`. Nunca versionar ese archivo.

## Ciclo de desarrollo

```powershell
venv\Scripts\python.exe -m unittest discover -s tests
venv\Scripts\python.exe -m unittest tests.test_prediction_committee
venv\Scripts\python.exe -m ruff check alttez scripts tests
venv\Scripts\python.exe scripts\apply_supabase_migrations.py
```

Las pruebas usan `unittest`. Ruff existe y CI lo ejecuta como job no bloqueante.
Las pruebas que requieren red o una base real no deben incorporarse por accidente
a la suite unitaria.

## Dashboard

```powershell
venv\Scripts\python.exe -m streamlit run alttez/dashboard/app.py
```

Separar calculos y transformaciones de Streamlit para probarlos sin navegador.

## Proveedores

```powershell
venv\Scripts\python.exe -m alttez.ingestion.provider_healthcheck --json
venv\Scripts\python.exe scripts\smoke_live_pipeline.py --date AAAA-MM-DD
```

El smoke sin `--write` es la opcion inicial. `--write` persiste datos.

Estados de salida del smoke: `0` exito con agenda, `3` consulta valida sin agenda
(`idle/no_schedule`) y `1` fallo operativo. El codigo `3` no representa un fallo
del modelo ni debe disparar una alerta de incidente.

## Migraciones

```powershell
# Validar, sin aplicar
venv\Scripts\python.exe scripts\apply_supabase_migrations.py

# Aplicar con SUPABASE_DB_URL configurada
venv\Scripts\python.exe scripts\apply_supabase_migrations.py --apply
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
alerta critica.

## Modelos

```powershell
venv\Scripts\python.exe scripts\run_challenger_cycle.py --date AAAA-MM-DD
venv\Scripts\python.exe scripts\fit_probability_calibration.py --help
venv\Scripts\python.exe scripts\evaluate_calibration.py --help
venv\Scripts\python.exe scripts\activate_model_version.py --help
venv\Scripts\python.exe scripts\rollback_active_model.py --help
```

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
