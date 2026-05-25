import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from alttez.features.elo import EloEngine
from alttez.ingestion.international import InternationalLoader


def main() -> None:
    # 1. Cargar y preparar datos.
    loader = InternationalLoader()
    df = loader.fetch_data()
    engine = EloEngine()
    df = engine.compute_elo_features(df)

    print("\n=== PREPARANDO MACHINE LEARNING ===")

    # 2. Definir features (X) y target (y).
    # 1 = gana el equipo local, 0 = empate o gana el visitante.
    df["target_team_a_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["is_neutral"] = df["neutral"].astype(int)

    X = df[["elo_difference", "is_neutral"]]
    y = df["target_team_a_win"]

    # 3. Separar en entrenamiento y prueba sin mezclar el orden temporal.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

    # 4. Entrenar XGBoost.
    print("Entrenando modelo XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        objective="binary:logistic",
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    # 5. Evaluar el modelo.
    preds_proba = model.predict_proba(X_test)[:, 1]
    preds_class = model.predict(X_test)

    acc = accuracy_score(y_test, preds_class)
    loss = log_loss(y_test, preds_proba)
    print(f"Accuracy (Precision): {acc:.2%}")
    print(f"Log-Loss (Calibracion): {loss:.4f} (Mas cerca a 0.60 es excelente)")

    # 6. Inferencia Mundial 2026: Colombia vs Brazil en cancha neutral.
    print("\n=== SIMULACION MUNDIAL 2026 ===")
    elo_colombia = engine.ratings["Colombia"]
    elo_brazil = engine.ratings["Brazil"]

    diferencia_elo = elo_colombia - elo_brazil

    cruce_mundial = pd.DataFrame(
        {
            "elo_difference": [diferencia_elo],
            "is_neutral": [1],
        }
    )

    prob_colombia = model.predict_proba(cruce_mundial)[0][1]
    prob_no_gana = model.predict_proba(cruce_mundial)[0][0]

    print(f"Enfrentamiento: Colombia ({int(elo_colombia)}) vs Brazil ({int(elo_brazil)})")
    print(f"Diferencia ELO: {diferencia_elo:.1f} puntos")
    print(f"Probabilidad Victoria Colombia: {prob_colombia:.1%}")
    print(f"Probabilidad Empate/Victoria Brazil: {prob_no_gana:.1%}")


if __name__ == "__main__":
    main()
