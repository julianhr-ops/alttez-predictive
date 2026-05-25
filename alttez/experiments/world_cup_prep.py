from alttez.features.elo import EloEngine
from alttez.ingestion.international import InternationalLoader


def main() -> None:
    # 1. Extraer los datos.
    loader = InternationalLoader()
    df = loader.fetch_data()

    # 2. Calcular ELO.
    engine = EloEngine()
    df_with_elo = engine.compute_elo_features(df)

    # 3. Ver los resultados mas recientes.
    print("\n=== RANKING ELO ALTTEZ ACTUAL (TOP 10) ===")
    current_rankings = sorted(engine.ratings.items(), key=lambda item: item[1], reverse=True)[:10]
    for rank, (team, elo) in enumerate(current_rankings, 1):
        print(f"{rank}. {team}: {int(elo)}")

    # 4. Ver un ejemplo reciente de Colombia.
    colombia_recent = df_with_elo[
        (df_with_elo["home_team"] == "Colombia") | (df_with_elo["away_team"] == "Colombia")
    ].tail(5)
    print("\n=== ULTIMOS 5 PARTIDOS DE COLOMBIA ===")
    print(
        colombia_recent[
            ["date", "home_team", "away_team", "home_score", "away_score", "elo_home_before", "elo_away_before"]
        ]
    )


if __name__ == "__main__":
    main()
