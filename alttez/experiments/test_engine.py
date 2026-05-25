import pandas as pd
from alttez.features.statistics import StatisticsEngine

# 1. Crear un dataset de mentira (Formato FootyStats)
data = {
    'Date': ['2026-05-01', '2026-05-08', '2026-05-15', '2026-05-22', '2026-05-29'],
    'Season': ['2026', '2026', '2026', '2026', '2026'],
    'Home': ['Nacional', 'DIM', 'Envigado', 'Nacional', 'DIM'],
    'Away': ['DIM', 'Envigado', 'Nacional', 'Envigado', 'Nacional'],
    'HG': [2, 1, 0, 3, 1], # Home Goals
    'AG': [1, 1, 2, 0, 1], # Away Goals
    'Result': ['H', 'D', 'A', 'H', 'D']
}

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])

print("=== DATOS CRUDOS ===")
print(df)

# 2. Inicializar el motor (Ventana de 2 partidos, Margen de 1 gol)
engine = StatisticsEngine(match_history_window=2, goal_diff_margin=1)

# 3. Pedirle que calcule las Rachas de Victorias Locales (HW) y Goles a Favor (HGF)
print("\n=== PROCESANDO ESTADISTICAS ===")
processed_df = engine.compute_stats(df, ['HW', 'HGF'])

print("\n=== DATOS CON FEATURE ENGINEERING ===")
print(processed_df[['Date', 'Home', 'Away', 'HW', 'HGF']])