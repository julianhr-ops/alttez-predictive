import numpy as np
from scipy.stats import poisson
import pandas as pd


def calcular_puntos_torneo(pred_h, pred_a, real_h, real_a, is_knockout=False):
    """Aplica estrictamente tu reglamento de puntuación."""
    mult = 2 if is_knockout else 1
    pts = 0

    # 1. Acertar Resultado: Ganador/Empate (5 pts)
    if np.sign(pred_h - pred_a) == np.sign(real_h - real_a):
        pts += 5 * mult

    # 2. Acertar Goles Equipo Local (2 pts)
    if pred_h == real_h:
        pts += 2 * mult

    # 3. Acertar Goles Equipo Visitante (2 pts)
    if pred_a == real_a:
        pts += 2 * mult

    # 4. Acertar Diferencia de Goles (1 pt)
    if (pred_h - pred_a) == (real_h - real_a):
        pts += 1 * mult

    return pts


def main():
    print("=== SIMULADOR DE VALOR ESPERADO (ALTTEZ) ===\n")

    # CASO DE PRUEBA: Partido cerrado donde Equipo A es ligeramente favorito.
    # Digamos que nuestro motor calculó que A marca 1.4 goles en promedio y B marca 0.9.
    lambda_home = 1.4
    lambda_away = 0.9
    print(f"Fuerza de Ataque -> Local: {lambda_home} xG | Visitante: {lambda_away} xG\n")

    # 1. Generar matriz de probabilidades de Poisson (Simular 10,000 futuros)
    max_goles = 5
    prob_matrix = np.zeros((max_goles + 1, max_goles + 1))

    for h in range(max_goles + 1):
        for a in range(max_goles + 1):
            # La probabilidad de que queden H-A goles
            prob_matrix[h, a] = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)

    # 2. Fuerza Bruta Matemática: Probar todos los pronósticos posibles
    resultados = []

    # Iteramos sobre todos los pronósticos que TÚ podrías enviar
    for pred_h in range(max_goles + 1):
        for pred_a in range(max_goles + 1):
            puntos_esperados = 0

            # Simulamos ese pronóstico contra todos los resultados reales posibles
            for real_h in range(max_goles + 1):
                for real_a in range(max_goles + 1):
                    probabilidad_de_ocurrencia = prob_matrix[real_h, real_a]
                    puntos_ganados = calcular_puntos_torneo(pred_h, pred_a, real_h, real_a, is_knockout=False)

                    # Fórmula del Valor Esperado: E[P] = Prob * Puntos
                    puntos_esperados += probabilidad_de_ocurrencia * puntos_ganados

            resultados.append({
                'Tu Pronostico': f"{pred_h} - {pred_a}",
                'Puntaje Esperado': puntos_esperados
            })

    # 3. Mostrar los resultados ordenados
    df_resultados = pd.DataFrame(resultados).sort_values(by='Puntaje Esperado', ascending=False)
    print("⚽ TOP 5 MEJORES PRONÓSTICOS (Fase de Grupos):")
    print(df_resultados.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
