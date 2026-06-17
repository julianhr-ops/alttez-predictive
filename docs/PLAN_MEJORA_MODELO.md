# Plan de mejora del modelo — ALTTEZ

> Basado en datos reales de prediction memory. Ventana analizada: 2026-05-26 → 2026-06-16. Última revisión: 2026-06-16.

## 1. Diagnóstico (evidencia verificada)

Consultas directas a `match_predictions` / `prediction_evaluations` / `prediction_runs` / `matches` / `raw_matches_api`.

### 1.1 Las métricas crudas eran un artefacto de medición

El "Brier 0.93 / acc 15%" inicial se debía a (a) solo 20 predicciones cerradas y (b) **45% de filas duplicadas** (626 filas, 346 `match_id`) que sobre-ponderaban forecasts viejos. Tras correr el sweep de cierre (`close_finished_prediction_evaluations`, +38 → 58 evaluados) y deduplicar:

| Set | n | Brier 1X2 | Acc 1X2 | MAE goles |
|---|---|---|---|---|
| Inicial (20, con duplicados) | 20 | 0.93 | 15% | 2.15 |
| Todos evaluados | 58 | 0.769 | 36% | 2.14 |
| **Dedup (1 por partido, último)** | **29** | **0.648** | **55%** | 2.07 |

Real ≈ Brier 0.648 (ya por debajo del azar 0.667), accuracy 55%. Mediocre, no catastrófico.

### 1.2 El sesgo estructural es real y persiste

Sobre el set dedup (29 partidos con etiqueta real):

| Clase | Predicho | Real |
|---|---|---|
| Home | **27 (93%)** | 14 (48%) |
| Draw | **0** | 6 (21%) |
| Away | 2 (7%) | 9 (31%) |

- λ_home prom **1.64** vs λ_away **1.06** → tilt local fijo de +0.59 que no aplica en sedes neutrales de torneo.
- Empates colapsados: el argmax sobre λ inflada **nunca** elige D.
- Visitante sistemáticamente sub-predicho. MAE goles 2.07 (alto).

### 1.3 Cuello de botella real: faltan etiquetas

- Solo **29 partidos** con resultado real evaluable. El sweep cerró todo lo cerrable.
- Los resultados del torneo **no están ingestados**: en `matches` las fechas recientes están como `scheduled` con marcador placeholder **0-0** (290 filas); `raw_matches_api` solo trae **2 finished**. El proveedor (football_data_org free) no entrega finales para estos fixtures.
- **No se puede backfillear lo que no existe.** Desbloquear el aprendizaje exige primero ingesta de resultados (R2 + cobertura de proveedor), no un flip de status.
- Contaminación: 387/626 friendlies junto a 194 world_cup.

## 2. Plan priorizado por ROI

### P0 · Cerrar el loop de datos (precondición) — PARCIAL
- ✅ **Sweep de cierre ejecutado**: `close_finished_prediction_evaluations(target_date=None)` → evaluados 20 → 58. Cerró todo lo que tenía resultado real.
- ✅ **Dedup como criterio de métrica**: agregar por `match_id` tomando la última predicción (no borrar filas: cada run es un forecast point-in-time legítimo de auditoría).
- ❌ **Backfill masivo BLOQUEADO**: los resultados del torneo no están ingestados (`matches` con 0-0 placeholder; raw con 2 finished). No se fabrican etiquetas.

### P0.5 · Ingesta de resultados — ✅ HECHO (multi-fuente)
- **Fuentes**: `football_data_org` (free, solo WC) + **ESPN** scoreboard (keyless, **cubre friendlies** — el hueco). api_football descartada: key muerta en host directo y RapidAPI.
- `alttez/ingestion/results_backfill.py`: funciones puras (`extract_finished_results`, `match_result_to_row`) + orquestador `backfill_finished_results(source=…)`, idempotente, dry-run por defecto. Mapea por par de equipos normalizado con tolerancia ±1 día (skew UTC/local).
- `alttez/ingestion/espn_results.py`: fetch + extractor ESPN (`fifa.friendly`, `fifa.world`), pure-function parser.
- `scripts/backfill_match_results.py` (CLI `--source espn|football_data`) + `tests/test_results_backfill.py` (8 tests).
- Ejecutado: WC (football_data, 16 filas) + ESPN 2026-05-26..16 (160 finished, 139 matched, 107 filas). **Evaluadas: 20 → 287.** Con probabilidad almacenada: 34 → **137**.
- **Pendiente recurrente**: agendar el backfill ESPN en el proceso diario (hoy manual). Opcional: reactivar key api_football para redundancia.

### P2 · Calibración de probabilidades — ❌ DESCARTADO POR DATOS
Probado con n=137 partidos etiquetados (leave-one-out): calibración 3-pesos, temperature scaling y draw-boost **todos empeoran** el Brier (0.566 → 0.569/0.589/0.583). Razón: las probabilidades **ya están en tasa base** (pred-mean H/D/A [0.534, 0.233, 0.233] vs real [0.547, 0.277, 0.175]). No hay nada que calibrar; el déficit es **resolución**, no calibración.

### P2' · Capa de decisión draw-aware (reemplazo de P2, opcional/seguro)
El síntoma "el modelo dice local casi siempre" es el **argmax**, no la probabilidad: argmax [118, 1, 18] vs real [75, 38, 24] — predice empate 1 de 137 veces aunque pD≈0.23 es sano.
- El brief usa `best_market`/`predicted_outcome` = clase #1, que casi nunca es D.
- Fix de presentación: mostrar el triple de probabilidad y/o marcar "empate probable" cuando pD esté dentro de un margen del top. No mejora Brier; mejora utilidad del brief. Bajo riesgo.

### P1 · Corregir sesgo local / resolución — AHORA EL #1
El lever real: el modelo no discrimina (WC Brier 0.630 ≈ azar 0.667). Probabilidades planas cerca de la media.
- Medir/atenuar ventaja local en sedes neutrales (argmax over-predice home 118 vs 75 real).
- Re-anclar λ al scoring observado; mejorar fuerzas de equipo (enlaza P4).
- Requiere las etiquetas que P0.5 ya está acumulando.

### P3 · Segmentar por competición
- Separar friendly de world_cup/qualifiers; ponderar bajo o excluir friendlies.
- Baseline de goles por tipo de torneo.

### P4 · Usar el propio torneo como prior
- Fuerzas empíricas ataque/defensa de equipos con partidos jugados; blend con `world_cup_historical_seeds.csv`.

### P5 · Re-habilitar gobernanza con métricas confiables
- Con loop cerrado, `challenger_cycle` compara Brier sobre datos reales (hoy decide sobre 20 evals ruidosos).

## 3. Criterios de éxito
Brier 1X2 **< 0.66** y bajando · accuracy 1X2 **> 40%** · distribución H/D/A dentro de **±10pp** de la real · MAE goles **< 1.4**.

## 4. Orden de ejecución
**P0 (✅) → P0.5 (✅) → ~~P2 calibración~~ (descartado por datos) → P1 (resolución/sesgo local, #1) → (P2' decisión draw-aware ∥ P4) → P5.**
Con 287 etiquetas (137 con prob) el dato manda: calibrar no ayuda; el salto real es resolución (P1) + más fuerzas de equipo (P4).

## 5. Estado de ejecución (2026-06-16)
- ✅ **P0**: sweep de cierre (+38). No destructivo.
- ✅ **P0.5**: backfill multi-fuente (football_data WC + **ESPN friendlies**). 123 filas actualizadas en total, 229 evals cerradas. **Evaluadas 20 → 287; con prob 34 → 137.** Suite 117 tests OK.
- ❌ **P2 calibración**: descartado con evidencia (LOO n=137 empeora Brier; probas ya en tasa base).
- Métricas reales: Brier global 0.634, acc 46.7%; WC 0.630 (≈ azar → poca resolución); friendly 0.557.
- **Siguiente**: P1 (sesgo local/resolución). Opcional barato: P2' (brief draw-aware). Recurrente: agendar backfill ESPN diario.
- P3–P5: pendientes.
