# Configuracion y secretos

## Precedencia

El proyecto carga `.env` mediante `alttez/api/settings.py`. La fuente versionada de
nombres y ejemplos es `.env.example`; el `.env` real es secreto local.

Al agregar una variable:

1. Definir un default seguro en codigo.
2. Agregarla a `.env.example` sin credenciales.
3. Documentarla aqui.
4. Propagarla al workflow solo si la automatizacion la necesita.
5. Probar ausencia, valor valido y valor invalido.

## Familias

| Familia | Ejemplos | Responsabilidad |
| --- | --- | --- |
| Base | `SUPABASE_DB_URL` | Conexion Postgres/Supabase |
| Historico | `ALTTEZ_DATA_SOURCE`, `ALTTEZ_SUPABASE_VIEW` | Fuente de datos |
| Proveedores | `ALTTEZ_PROVIDER_STACK`, `ALTTEZ_LIVE_PROVIDER_STACK` | Orden |
| ESPN | `ALTTEZ_ESPN_*` | Agenda keyless y ligas |
| football-data.org | `ALTTEZ_FOOTBALL_DATA_*` | Provider estructurado |
| Local | `ALTTEZ_STATSBOMB_OPEN_ROOT`, `ALTTEZ_LOCAL_JSON_PATH` | Fuentes locales |
| Alertas | `ALTTEZ_ALERT_*`, `ALTTEZ_ALERTS_TELEGRAM_*` | Incidentes |
| Brief | `ALTTEZ_DAILY_BRIEF_*` | Canal, credenciales y timeouts |
| Gobierno | `ALTTEZ_AUTO_PROMOTE_MODELS`, `ALTTEZ_SUPPORTED_PRODUCTION_MODEL_FAMILIES` | Promocion |
| Expansion | `ALTTEZ_EXPANSION_*` | Umbrales de cobertura |
| Modelo | `ALTTEZ_*_PATH`, `ALTTEZ_ACTIVE_MODEL_*` | Artefactos y seleccion |

La tabla agrupa variables; no reemplaza `.env.example` ni una busqueda de
`os.getenv` en codigo.

## Reglas de seguridad

- Nunca mostrar `.env` en logs, respuestas o commits.
- No incluir tokens o URLs con password en errores.
- Tratar bot tokens, chat IDs, webhooks y URL de base como secretos.
- Usar secrets de GitHub; no valores literales en YAML.
- Un healthcheck informa configurado/no configurado, no revela el valor.
- Un agente puede leer `.env.example`; no necesita abrir `.env`.

## Flags sensibles

- `ALTTEZ_AUTO_PROMOTE_MODELS`: puede cambiar el modelo activo.
- `ALTTEZ_ALLOW_LOCAL_FALLBACK`: altera la procedencia del historico.
- `ALTTEZ_DAILY_BRIEF_DELIVER`: habilita entrega externa.
- `ALTTEZ_ALERTS_TELEGRAM_ENABLED`: habilita alertas externas.
- `--write`, `--apply`, activacion y rollback: escriben estado persistente.
