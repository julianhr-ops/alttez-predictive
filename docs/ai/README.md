# Contexto del proyecto para IA

Esta carpeta es la fuente central de contexto para asistentes de IA, sin depender
de Claude, Codex, Copilot u otro proveedor. Su objetivo es permitir cambios seguros
sin tener que reconstruir la arquitectura en cada sesion.

## Lectura recomendada

1. [`../../AGENTS.md`](../../AGENTS.md): reglas operativas y contrato minimo.
2. [`architecture.md`](architecture.md): componentes, dependencias y flujo de datos.
3. [`workflows.md`](workflows.md): instalacion, pruebas y procesos operativos.
4. [`data-and-models.md`](data-and-models.md): datos, inferencia, evaluacion y
   gobierno de modelos.
5. [`configuration.md`](configuration.md): familias de variables de entorno,
   secretos y precedencia.
6. [`maintenance.md`](maintenance.md): como mantener este contexto sincronizado.

## Resumen ejecutivo

ALTTEZ es un motor de prediccion de partidos de futbol orientado a selecciones y
torneos. Obtiene agenda y resultados desde una cadena de proveedores, normaliza e
ingiere datos, construye features ELO/Poisson/Klement, produce probabilidades 1X2 y
marcadores, evalua modelos, conserva memoria de predicciones y genera reportes
diarios. Una aplicacion Streamlit presenta analisis y monitoreo.

El flujo productivo principal es:

```text
proveedores -> ingestion/certificacion -> historico
            -> features -> router/comite -> probabilidades y marcador
            -> memoria/evaluacion -> brief/dashboard/alertas

historico -> entrenamiento challenger -> evaluacion/gobierno
          -> registro y artefactos -> router/comite
```

## Fuentes de verdad

| Tema | Fuente canonica |
| --- | --- |
| Dependencias | `requirements.txt` |
| Configuracion de ejemplo | `.env.example` |
| CI y version de Python | `.github/workflows/ci.yml` |
| Proceso diario | `.github/workflows/daily_process.yml`, `scripts/run_daily_process.ps1` |
| Esquema de features | `alttez/features/klement.py` |
| Seleccion del predictor | `alttez/inference/router.py` |
| Ensamble | `alttez/inference/prediction_committee.py` |
| Registro y memoria | `alttez/operations/prediction_memory.py` |
| Esquema SQL | `sql/supabase/migrations/` |
| Contratos de ingestion | `alttez/ingestion/contracts.py` |
| Contrato de pruebas | `tests/` y `unittest` |

## Limites de esta documentacion

- No contiene credenciales, valores reales de `.env` ni datos personales.
- No duplica la lista completa de funciones; para eso se consulta el codigo.
- `README.md` es la guia para personas usuarias y operacion general.
  `docs/ai/` explica restricciones y relaciones necesarias para modificar el
  sistema.
- `.claude/` puede contener configuracion especifica de una herramienta, pero no es
  la fuente compartida de arquitectura.
