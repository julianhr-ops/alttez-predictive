# Auditoria tecnica integral de ALTTEZ

Fecha: 2026-08-03
Alcance: arquitectura, funcionalidad, datos, modelos, operacion, calidad,
seguridad, pruebas, dependencias y archivos potencialmente obsoletos.

## Resumen ejecutivo

ALTTEZ es un sistema funcional y ambicioso de prediccion de futbol internacional.
Integra proveedores externos, Supabase/Postgres, modelos estadisticos y ML,
gobierno champion/challenger, reportes, alertas y una interfaz Streamlit.

La arquitectura conceptual es razonable y contiene controles maduros: fallbacks
explicitos, migraciones incrementales, memoria de predicciones, calibracion,
metricas probabilisticas y pruebas para la mayor parte del dominio.

Su principal riesgo no es la ausencia de funcionalidad, sino la acumulacion de
capas:

- modulos y funciones excesivamente grandes;
- dos generaciones de modelado conviviendo;
- dependencias sin clasificar entre directas y transitivas;
- un reemplazo local e incompleto de Pydantic;
- excepciones amplias que convierten fallos operativos en datos vacios;
- CI que tolera errores de lint;
- artefactos joblib/pickle cargados como codigo confiable;
- scripts y datos experimentales fuera del flujo productivo.

Evaluacion general:

| Area | Estado |
| --- | --- |
| Funcionalidad de dominio | Solida |
| Arquitectura conceptual | Buena, con limites erosionados |
| Mantenibilidad | Riesgo medio-alto |
| Pruebas | Amplias, pero sin cobertura medida y con fragilidad de entorno |
| Operacion y observabilidad | Buena intencion, manejo de errores inconsistente |
| Seguridad | Riesgo medio; artefactos serializados requieren endurecimiento |
| Higiene del repositorio | Mejorable |
| Documentacion | Mejorada con `docs/ai/`; el legado aun diverge |

## Que hace el proyecto

1. Consulta agenda y resultados desde ESPN, football-data.org, StatsBomb Open Data,
   JSON local y compatibilidad opcional con API-Football.
2. Normaliza equipos, competiciones, estados, fechas y payloads de estadisticas.
3. Certifica calidad y persiste partidos/catalogos en Supabase.
4. Construye perfiles historicos, ELO, forma reciente y features Klement.
5. Genera probabilidades 1X2 y matrices de marcador con Poisson, Dixon-Coles,
   Poisson bivariado, modelos jerarquicos y XGBoost.
6. Combina predictores en un comite con pesos dependientes del rendimiento.
7. Calibra probabilidades y optimiza picks de marcador para reglas de puntuacion.
8. Guarda la prediccion tal como existia antes del partido y la evalua despues.
9. Entrena challengers, compara metricas y controla promocion/rollback.
10. Genera brief diario, simulacion del Mundial 2026, alertas y dashboard.

## Como lo hace

```text
Providers
  -> contratos y normalizacion
  -> certificacion
  -> Supabase / historico local permitido
  -> features
  -> router del modelo activo
  -> comite y calibracion
  -> prediccion, memoria y evaluacion
  -> dashboard, brief y alertas

Historico y evaluaciones
  -> challenger cycle
  -> artefacto joblib
  -> model_versions / model_artifacts
  -> seleccion productiva
```

Tecnologias principales:

- Python 3.13.
- pandas, NumPy y SciPy para datos/calculo.
- scikit-learn y XGBoost para ML.
- PyMC/ArviZ para modelo jerarquico.
- SQLAlchemy y psycopg 3 para Postgres/Supabase.
- Streamlit y Altair para interfaz.
- PowerShell y GitHub Actions para operacion.
- `unittest` y Ruff para calidad.

## Fortalezas confirmadas

- La inferencia tiene fallback Poisson y expone razon del fallback.
- Train e inference comparten `MODEL_FEATURE_COLUMNS`.
- Las migraciones son numeradas, progresivas y validadas en CI.
- Existe memoria point-in-time de predicciones y cierre posterior de resultados.
- El gobierno no depende solo de accuracy: usa Brier, Log Loss, RPS y calibracion.
- La promocion automatica esta apagada por defecto.
- El stack de proveedores aisla fallos y conserva alternativas.
- Los queries observados usan parametros SQLAlchemy, no concatenacion de entrada.
- `.env` y artefactos estan ignorados por Git.
- Hay 242 pruebas descubiertas y cobertura conceptual de casi todos los subsistemas.
- El proceso diario diferencia etapas fatales y degradaciones no fatales.

## Hallazgos criticos y altos

### A1. Implementacion local de `pydantic` suplanta la libreria real

Severidad: alta.

Evidencia:

- `pydantic/__init__.py` implementa 300+ lineas de una API parcial.
- `alttez/ingestion/contracts.py:6` importa `pydantic`, que resuelve al paquete
  local por precedencia del repositorio.
- `requirements.txt` no declara Pydantic.

Riesgos:

- semantica distinta a Pydantic sin que el consumidor lo sepa;
- validacion incompleta de `None`, unions, aliases y estructuras anidadas;
- incompatibilidad futura con librerias que importen Pydantic real;
- falsa sensacion de seguridad en contratos de ingestion;
- mantenimiento interno de una libreria de validacion.

Accion:

1. Declarar `pydantic` real como dependencia directa.
2. Ejecutar pruebas de contratos contra Pydantic v2.
3. Corregir diferencias de validacion deliberadamente.
4. Eliminar o renombrar el paquete local.

### A2. Carga de artefactos `joblib`/`pickle` sin frontera criptografica

Severidad: alta si la base o ruta de artefactos puede ser alterada por terceros.

Evidencia:

- `alttez/inference/router.py:461` usa `joblib.load`.
- `alttez/inference/prediction_committee.py:101` usa `joblib.load`.
- `alttez/models/model.py:126` usa `pickle.load`.
- `model_artifacts` almacena SHA-256, pero la descarga no verifica el hash antes de
  deserializar.

Joblib y pickle pueden ejecutar codigo durante deserializacion. Un hash almacenado
junto al mismo blob detecta corrupcion accidental, no autenticidad.

Accion:

- limitar escritura de `model_artifacts` a un rol de entrenamiento;
- verificar tamaño y SHA-256 en descarga;
- usar firma/HMAC administrada fuera de la misma tabla si el riesgo lo exige;
- validar familia, schema version y clases permitidas antes de activar;
- eliminar la ruta pickle heredada si no tiene consumidores.

### A3. Modulos monoliticos y funciones gigantes

Severidad: alta para mantenibilidad y regresiones.

Evidencia:

- `dashboard/app.py`: 1.852 lineas; `main`: 382.
- `ingestion/live_matches.py`: 1.447 lineas.
- `operations/prediction_memory.py`: 865 lineas.
- `training/challenger_cycle.py`: 803 lineas.
- `inference/router.py`: 798 lineas.
- `render_daily_brief_html`: 540 lineas.
- `build_brand_css`: 305 lineas.
- `build_klement_feature_state`: 249 lineas.

Esto contradice la politica modular documentada y mezcla transformacion, acceso a
datos, decision, formato y presentacion.

Accion:

- separar adaptadores de provider, repositorios SQL, servicios de dominio y UI;
- convertir HTML/CSS en templates/recursos dedicados;
- extraer subflujos puros del challenger y Klement;
- mantener fachadas compatibles para reducir el riesgo de migracion.

### A4. Fallos operativos pueden convertirse silenciosamente en “sin datos”

Severidad: alta para observabilidad y decisiones.

`prediction_memory.py` captura `Exception` en numerosas lecturas y devuelve
`DataFrame()` o `None`. El dashboard no siempre puede distinguir:

- tabla realmente vacia;
- migracion faltante;
- credenciales incorrectas;
- timeout;
- query incompatible;
- indisponibilidad de Supabase.

La resiliencia es correcta para mantener la UI viva, pero el resultado necesita un
estado tipado con `available`, `data`, `error_code` y `detail` seguro.

Accion:

- capturar excepciones esperadas por tipo;
- registrar con contexto sin secretos;
- devolver resultados con estado;
- reservar fallback vacio para capas de presentacion.

### A5. CI permite lint roto

Severidad: media-alta.

`.github/workflows/ci.yml:36` usa `continue-on-error: true`. Ruff reporta actualmente
cinco errores:

1. variable `model` sin uso en `features/hierarchical.py`;
2. import sin uso en `inference/router.py`;
3. argumento `raw_score` calculado pero no usado en HTML;
4. import `os` sin uso en `fit_dixon_coles_rho.py`;
5. import `IsotonicCalibration` sin uso.

Accion: corregir los cinco, hacer Ruff bloqueante y agregar `ruff format --check` o
un formatter acordado.

## Hallazgos medios

### M1. `requirements.txt` mezcla dependencias directas y transitivas

Hay paquetes sin referencias en codigo: Alembic, Boruta, matplotlib, seaborn, SHAP,
openpyxl, pyarrow, RapidFuzz y Levenshtein, entre otros. Starlette/Uvicorn tampoco
tienen una aplicacion ASGI en el repositorio.

Selenium y lxml aparecen solo en el scraper heredado de FootyStats. Su permanencia
depende de conservar ese modulo.

Accion:

- definir dependencias directas en `pyproject.toml`;
- generar un lock reproducible separado;
- crear extras `dashboard`, `bayes`, `scraping` y `dev`;
- retirar transitivas de la lista mantenida manualmente.

### M2. El proyecto no esta empaquetado

No hay `pyproject.toml`. Muchos scripts insertan la raiz en `sys.path`, lo que
duplica bootstrap y hace que el comportamiento dependa del directorio.

Accion: empaquetar `alttez`, agregar entry points y ejecutar scripts con
`python -m` o comandos instalados.

### M3. Motores y APIs heredadas conviven con el pipeline actual

Candidatos fuertes a archivar:

- `alttez/scraping/scraper.py`: sin consumidores encontrados; Selenium/FootyStats.
- `alttez/models/logistic.py`: sin consumidores encontrados.
- `alttez/models/extremeboosting.py`: distinto del XGBoost productivo del router.
- `alttez/experiments/test_engine.py`.
- `alttez/experiments/test_polla.py`: prueba manual dentro del paquete.
- `alttez/experiments/world_cup_prep.py`.
- `alttez/experiments/train_world_cup.py`: se declara experimental.

`models/model.py`, `training/trainer.py` y `training/tuner.py` forman una capa
antigua aun cubierta por pruebas. Debe decidirse si es un laboratorio soportado o
si se archiva como conjunto.

### M4. Scripts y CSV legacy en la raiz

Presentes localmente pero ignorados:

- `test_fbref_extraction.py`;
- `transform_data_poisson.py`;
- `mundial_2026_fixtures.csv`;
- `mundial_2026_team_stats.csv`;
- `poisson_team_metrics.csv`;
- `alttez_predictions_output.csv`.

Los dos scripts muestran mojibake/Unicode inconsistente y duplican flujos que ahora
tienen implementaciones bajo `scripts/` y `alttez/`.

Accion: mover ejemplos necesarios a `examples/` con pruebas, o eliminar copias
locales. No convertir CSV generados en fuentes canonicas.

### M5. Directorios locales sin clasificacion clara

- `archived_files/` esta sin seguimiento y no esta en `.gitignore`.
- `tmptvhsjedb/` esta ignorado, pero su nombre no explica procedencia.
- `venv/`, caches y artefactos conviven dentro de la raiz de trabajo.

Accion: eliminar residuos confirmados, ignorar explicitamente archivos de archivo
local o moverlos fuera del repositorio.

### M6. Pruebas pesadas mezcladas con pruebas unitarias

`tests/test_hierarchical_bayes.py:155` ejecuta PyMC con 300 draws, 300 tune y dos
cadenas. La suite completa tardo alrededor de un minuto en el entorno auditado y
requiere multiprocessing.

Accion:

- separar `unit`, `integration`, `slow` y `network`;
- usar una prueba determinista pequena para contrato;
- ejecutar la prueba estadistica completa en un job dedicado;
- fijar `cores=1` en entornos Windows restringidos cuando corresponda.

### M7. No hay cobertura, type checking ni auditoria de seguridad en CI

CI ejecuta unittest, migraciones y Ruff no bloqueante. No mide coverage, no ejecuta
Pyright/mypy, no audita dependencias y no detecta secretos.

Accion gradual:

1. Ruff bloqueante.
2. Coverage con umbral inicial basado en la realidad.
3. `pip-audit`/Dependabot.
4. detector de secretos.
5. type checker por modulos nuevos y luego expansion.

### M8. Recursos SQLAlchemy recreados con frecuencia

`get_engine()` crea un engine por llamada. Muchas funciones de lectura lo invocan
individualmente. Aunque los context managers cierran conexiones, crear pools
repetidamente desperdicia recursos y complica su ciclo de vida.

Accion: inyectar un engine/repository por proceso o cachearlo con una estrategia
explicita y `dispose()` al terminar.

### M9. Consenso y lambdas no representan exactamente el mismo modelo

El comite mezcla matrices ponderadas para 1X2/marcadores, pero expone `lambda_h` y
`lambda_a` como media simple de miembros soportados. Esas lambdas pueden no generar
la matriz consensuada y pueden confundir reportes de xG.

Accion: etiquetar las lambdas como resumen de miembros, ponderarlas igual que la
matriz o derivar un equivalente documentado.

### M10. Configuracion `.env` implementada manualmente

`api/settings.py` soporta lineas `KEY=VALUE` simples, pero no toda la semantica de
dotenv. El PowerShell diario implementa otro parser. Pueden divergir con comentarios,
escapes o valores complejos.

Accion: usar una libreria comun o definir y probar formalmente el subconjunto
soportado.

## Calidad de pruebas observada

Resultado en este entorno:

- descubiertas: 242;
- completadas sin assertion failure visible: 226;
- errores: 16;
- 15 errores asociados a escritura/limpieza de `TemporaryDirectory` bloqueada por
  el sandbox;
- 1 error asociado a `multiprocessing.Pipe` de PyMC bloqueado por el sandbox.

Esto no demuestra 16 defectos del producto. Si demuestra que la suite no es
portable a un entorno Windows restringido. En Linux CI puede pasar normalmente,
pero se necesita consultar el ultimo job real para afirmarlo.

No existe medicion de cobertura, por lo que cantidad de pruebas no equivale a un
porcentaje conocido.

## Archivos: clasificacion recomendada

### Mantener como nucleo

- `alttez/api`, `ingestion`, `features`, `inference`, `operations`, `reports`,
  `dashboard`;
- migraciones SQL;
- scripts usados por `run_daily_process.ps1`;
- pruebas asociadas;
- configuracion del Mundial y graficos realmente consumidos.

### Mantener, pero refactorizar

- `dashboard/app.py`;
- `ingestion/live_matches.py`;
- `operations/prediction_memory.py`;
- `training/challenger_cycle.py`;
- `inference/router.py`;
- `features/klement.py`;
- `reports/daily_brief/render_html.py`.

### Revisar uso y archivar si no hay propietario

- `alttez/scraping/`;
- clasificadores de `alttez/models/`;
- `training/trainer.py` y `training/tuner.py`;
- experimentos que no alimentan produccion;
- `storage/network/browser.json`, `leagues.json`, `models.json`;
- `Prophitbet.iml`;
- `open_alttez_dashboard.cmd` frente al script PowerShell equivalente.

### Eliminar del workspace despues de confirmar

- descargas y locks bajo `archived_files/`;
- carpetas temporales;
- scripts raiz ignorados con mojibake;
- CSV generados que puedan regenerarse.

## Plan de saneamiento

### Fase 0: controles inmediatos

- Corregir Ruff y hacerlo bloqueante.
- Sustituir el Pydantic local.
- Verificar hash/tamano de artefactos antes de `joblib.load`.
- Clasificar suite slow/integration.
- Confirmar que CI principal esta verde.

### Fase 1: higiene y dependencias

- Crear `pyproject.toml`.
- Separar dependencias directas, extras y lock.
- Archivar/eliminar codigo sin consumidores.
- Limpiar raiz y `.gitignore`.
- Corregir documentacion legacy.

### Fase 2: limites arquitectonicos

- Extraer providers de `live_matches.py`.
- Crear repositorios para memoria, modelos y agenda.
- Dividir dashboard por vistas/controladores.
- Extraer templates HTML/CSS.
- Introducir resultados tipados para disponibilidad/fallback.

### Fase 3: calidad sistematica

- Coverage y type checking incremental.
- Tests de integracion con Postgres efimero.
- Contratos de artefactos con schema version.
- Auditoria de dependencias y secretos.
- Metricas de complejidad y presupuesto por modulo.

## Conclusion

El proyecto no es un prototipo vacio: contiene un pipeline real y decisiones de
modelado/operacion por encima del promedio. Tampoco esta listo para crecer sin
saneamiento. La prioridad debe ser reducir ambiguedad: una sola libreria de
contratos, una sola generacion de modelos soportada, dependencias declaradas,
errores observables y modulos con responsabilidades delimitadas.

No se recomienda una reescritura. Se recomienda una migracion incremental,
protegida por las pruebas existentes y por contratos mas estrictos.
