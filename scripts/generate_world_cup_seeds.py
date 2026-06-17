"""Genera world_cup_historical_seeds.csv: priores de fuerza para los 48 equipos
del Mundial 2026, derivados del historico internacional (martj42/international_results).

Estos sembrados son el segundo eslabon de la cadena de contingencia de arranque en frio
descrita en CLAUDE.md (Data del Mundial Actual -> Semillas Historicas -> Neutro 1.0).
Solo se consultan cuando un equipo no tiene metricas del torneo actual, por lo que
representan una fuerza historica estable, no una estimacion de forma reciente.

Escala identica a poisson_team_metrics.csv:
    fuerza_ataque  = promedio_goles_anotados_equipo  / promedio_goles_global
    fuerza_defensa = promedio_goles_recibidos_equipo / promedio_goles_global
Un valor 1.0 es exactamente neutral (la media del dataset).

Las semillas se escriben con la grafia del fixture (home_team/away_team de
mundial_2026_fixtures.csv) para que la busqueda O(1) de predict_matches.py acierte,
aunque la fuerza se calcula sobre el nombre canonico del historico.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from alttez.api.settings import load_local_env
from alttez.ingestion.international import InternationalLoader
from alttez.ingestion.live_matches import canonical_team_name

FIXTURES_PATH = Path("mundial_2026_fixtures.csv")
SEEDS_PATH = Path("world_cup_historical_seeds.csv")


def load_world_cup_teams(fixtures_path: Path = FIXTURES_PATH) -> list[str]:
    """Devuelve los nombres de equipo (grafia del fixture) que disputan el Mundial."""
    fixtures = pd.read_csv(fixtures_path)
    teams = pd.concat([fixtures["home_team"], fixtures["away_team"]], ignore_index=True)
    return sorted({str(team).strip() for team in teams.dropna() if str(team).strip()})


def compute_strength_table(historical_df: pd.DataFrame, *, since_year: int | None = None) -> pd.DataFrame:
    """Tabla equipo_canonico -> fuerza_ataque/fuerza_defensa sobre el historico.

    Normaliza por la media global de goles por equipo/partido, de modo que la
    escala coincide con poisson_team_metrics.csv (neutral = 1.0).
    """
    df = historical_df.dropna(subset=["home_score", "away_score"]).copy()
    if since_year is not None:
        years = pd.to_datetime(df["date"], errors="coerce").dt.year
        df = df[years >= since_year]
    if df.empty:
        raise RuntimeError("Historico vacio tras el filtrado; no se pueden derivar sembrados.")

    home = df[["home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "scored", "away_score": "conceded"}
    )
    away = df[["away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "scored", "home_score": "conceded"}
    )
    long = pd.concat([home, away], ignore_index=True)
    long["team"] = long["team"].astype(str).str.strip()
    long = long[long["team"] != ""]

    global_avg = long["scored"].mean()
    if not pd.notna(global_avg) or global_avg <= 0:
        raise RuntimeError("Promedio global de goles invalido; no se pueden derivar sembrados.")

    grouped = long.groupby("team").agg(
        partidos=("scored", "size"),
        prom_anotados=("scored", "mean"),
        prom_recibidos=("conceded", "mean"),
    )
    grouped["fuerza_ataque"] = (grouped["prom_anotados"] / global_avg).round(4)
    grouped["fuerza_defensa"] = (grouped["prom_recibidos"] / global_avg).round(4)
    return grouped


def build_seed_rows(
    teams: list[str],
    known_names: set[str],
    strength_table: pd.DataFrame,
    *,
    min_matches: int = 10,
) -> tuple[list[dict], list[str]]:
    """Mapea cada equipo del fixture a su fuerza historica.

    Devuelve (filas, no_resueltos). Un equipo sin historico suficiente se omite:
    para esos casos predict_matches.py ya cae al neutro estricto 1.0, que es el
    comportamiento correcto cuando no hay senal historica fiable.
    """
    rows: list[dict] = []
    unresolved: list[str] = []
    for team in teams:
        canonical = canonical_team_name(team, known_names)
        if canonical not in strength_table.index:
            unresolved.append(team)
            continue
        stats = strength_table.loc[canonical]
        if int(stats["partidos"]) < min_matches:
            unresolved.append(f"{team} (solo {int(stats['partidos'])} partidos)")
            continue
        rows.append(
            {
                "equipo": team,
                "equipo_canonico": canonical,
                "partidos_historicos": int(stats["partidos"]),
                "fuerza_ataque": float(stats["fuerza_ataque"]),
                "fuerza_defensa": float(stats["fuerza_defensa"]),
            }
        )
    return rows, unresolved


def write_seeds(rows: list[dict], path: Path = SEEDS_PATH) -> None:
    pd.DataFrame(rows).sort_values("equipo").to_csv(path, index=False)


def main() -> int:
    load_local_env(override=True)
    parser = argparse.ArgumentParser(description="Genera sembrados historicos del Mundial 2026.")
    parser.add_argument("--fixtures", default=str(FIXTURES_PATH))
    parser.add_argument("--output", default=str(SEEDS_PATH))
    parser.add_argument("--since-year", type=int, default=None, help="Limita el historico a este anio en adelante.")
    parser.add_argument("--min-matches", type=int, default=10, help="Minimo de partidos para sembrar (sino, neutro).")
    args = parser.parse_args()

    teams = load_world_cup_teams(Path(args.fixtures))
    print(f"Equipos del Mundial detectados: {len(teams)}")

    historical_df = InternationalLoader().fetch_data()
    known_names = set(historical_df["home_team"].dropna()).union(set(historical_df["away_team"].dropna()))
    print(f"Partidos historicos cargados: {len(historical_df)}")

    strength_table = compute_strength_table(historical_df, since_year=args.since_year)
    rows, unresolved = build_seed_rows(teams, known_names, strength_table, min_matches=args.min_matches)

    write_seeds(rows, Path(args.output))
    print(f"Sembrados escritos: {len(rows)} en '{args.output}'")
    if unresolved:
        print(f"Sin sembrar (caen a neutro 1.0): {len(unresolved)}")
        for item in unresolved:
            print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
