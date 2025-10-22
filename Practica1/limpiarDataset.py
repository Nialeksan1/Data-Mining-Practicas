import pandas as pd
from datetime import datetime

# Configuración global
file_path = "../Ingreso, Gasto y Financiamiento Público ( 2011 - Actual ) de SHCP.csv"
output_path = "../Ingreso_Gasto_Financiamiento_Pena_AMLO_2012-2024.csv"

# Cargar dataset
df = pd.read_csv(file_path, low_memory=False)
print(f"Dataset original cargado: {df.shape[0]:,} registros.")

# Verificar columnas del dataset (para confirmar estructura)
print("Columnas detectadas:", df.columns.tolist())

# Crear columna 'fecha' para filtrado preciso
mes_map = {
    'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
    'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
}
df['mes_num'] = df['mes'].map(mes_map)
if df['mes_num'].isna().any():
    raise ValueError("Algunos meses no se mapearon correctamente. Revisar los valores en la columna 'mes'.")
df['fecha'] = pd.to_datetime(df['ciclo'].astype(str) + '-' + df['mes_num'].astype(str) + '-01')

# Filtrar por rango de fechas: 1 de diciembre de 2012 a 30 de septiembre de 2024
start_date = pd.to_datetime('2012-12-01')
end_date = pd.to_datetime('2024-09-30')
df_filtered = df[(df['fecha'] >= start_date) & (df['fecha'] <= end_date)].copy()

# Verificación
print(f"Registros filtrados: {df_filtered.shape[0]:,} (de {df.shape[0]:,} totales).")
print(f"Rango de fechas: {df_filtered['fecha'].min().strftime('%Y-%m-%d')} a {df_filtered['fecha'].max().strftime('%Y-%m-%d')}")

# Guardar dataset filtrado
df_filtered.to_csv(output_path, index=False)
print(f"Dataset filtrado guardado en {output_path}")

# Mostrar resumen básico
desc = df_filtered[['ciclo', 'mes', 'monto', 'subtema', 'sector']].describe(include='all')
print("\nResumen básico del dataset filtrado:")
print(desc.to_string(float_format='{:.2f}'.format))