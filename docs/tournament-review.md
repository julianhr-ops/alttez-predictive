# Cierre de torneo

ALTTEZ puede evaluar un torneo usando exclusivamente las predicciones guardadas en
`public.match_predictions` y los resultados cerrados en
`public.prediction_evaluations`.

## Generar el informe

```powershell
venv\Scripts\python.exe scripts\generate_tournament_review.py `
  --tournament "FIFA World Cup 2026" `
  --date-from 2026-06-11 `
  --date-to 2026-07-19
```

La conexion se toma de `SUPABASE_DB_URL`. Tambien puede pasarse con
`--database-url`. Las salidas quedan por defecto en:

```text
artifacts/tournament-reviews/<torneo>-review.json
artifacts/tournament-reviews/<torneo>-review.html
```

## Contenido

- cobertura de predicciones persistidas y evaluables;
- control temporal de prediccion frente a kickoff;
- accuracy 1X2;
- Brier Score, Log Loss, RPS y ECE;
- marcador exacto y error absoluto de goles;
- puntos obtenidos y maximos bajo reglas de polla;
- tasa de fallback;
- desglose por version del modelo;
- desglose por fase;
- detalle partido por partido;
- valoracion tecnica de 0 a 100;
- confianza de la muestra y advertencias.

## Politica temporal

- `valid_pre_match`: creada antes del kickoff; es evaluable.
- `invalid_after_kickoff`: creada durante o despues; se excluye.
- `unverifiable`: falta kickoff o timestamp; puede incluirse si tiene resultado,
  pero el informe declara la limitacion.

Una prediccion regenerada despues del torneo nunca se presenta como historica.

## Valoracion

| Componente | Peso |
| --- | ---: |
| Calidad probabilistica, Brier y RPS | 40% |
| Acierto 1X2 | 30% |
| Calibracion ECE | 15% |
| Marcador exacto y error de goles | 15% |

La cobertura no se oculta dentro del score. Se presenta por separado junto con una
confianza de muestra `Alta`, `Media` o `Baja`. Esto evita comparar un score basado
en pocos partidos con otro que cubre el torneo completo.

La metodologia y sus pesos tambien se incluyen en el JSON para que cada informe
sea auditable.

## Limitacion actual

El reporte evalua predicciones por partido. La comparacion del cuadro completo
pronosticado frente al cuadro real requiere conservar un snapshot pre-torneo del
bracket y de las probabilidades de avance. Si ese artefacto existe, puede agregarse
como segunda seccion sin alterar las metricas por partido.

## Entrega automatica por Telegram

Los torneos que deben producir cierre se registran en
`storage/tournament_reviews.json`. Cada entrada declara:

- una clave idempotente;
- nombre visible y texto de busqueda en la memoria;
- rango oficial de fechas;
- fecha minima de entrega;
- cantidad minima de partidos;
- cobertura evaluable minima.

El proceso diario ejecuta, despues del backfill de resultados:

```powershell
venv\Scripts\python.exe scripts\process_tournament_reviews.py --date AAAA-MM-DD --deliver
```

La tabla `public.tournament_review_deliveries` registra el resultado. Una clave con
estado `delivered` no vuelve a enviarse. `--force` existe solo para reenvios
operativos deliberados.

El transporte reutiliza `ALTTEZ_DAILY_BRIEF_TELEGRAM_BOT_TOKEN` y
`ALTTEZ_DAILY_BRIEF_TELEGRAM_CHAT_ID`. El cierre se entrega como HTML adjunto con
valoracion, 1X2, Brier y cobertura en el caption.

Para agregar un torneo nuevo no se modifica codigo: se agrega una entrada al
registro con su calendario oficial y gates de cobertura verificables.
