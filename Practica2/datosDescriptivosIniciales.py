import pandas as pd
import numpy as np
from datetime import datetime  # Para timestamp
from collections import defaultdict  # Para agrupar tipos

file_path = "../Ingreso_Gasto_Financiamiento_Pena_AMLO_2012-2024.csv"

# Configuración global
pd.set_option('display.float_format', lambda x: '%.0f' % x)
pd.set_option('display.max_colwidth', 50)  # Limita ancho de columnas para legibilidad

# Timestamp para trazabilidad
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("=" * 80)
print("[+] ANÁLISIS DESCRIPTIVO: Ingreso, Gasto y Financiamiento Público de Peña y AMLO")
print(f"Ejecutado: {timestamp} | Registros: {pd.read_csv(file_path, nrows=1).shape[1]} columnas esperadas")
print("=" * 80)

# 1. CARGA Y EXPLORACIÓN INICIAL
print("\n1. CARGA Y EXPLORACIÓN INICIAL")
print("-" * 50)

df = pd.read_csv(file_path, low_memory=False)

print("✓ Dataset cargado exitosamente")
print(f"  • Forma: {df.shape[0]:,} filas × {df.shape[1]} columnas")
print(f"  • Memoria aproximada: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

print("\n[+] Tipos de datos (agrupados por tipo):")
dtype_cols = defaultdict(list)
for col, dtype in df.dtypes.items():
    dtype_cols[str(dtype)].append(col)
for dtype in sorted(dtype_cols.keys()):
    cols_str = ', '.join(dtype_cols[dtype])
    print(f"  {dtype}: {cols_str}")

print("\n[+] Valores nulos por columna:")
nulls = df.isnull().sum()
if nulls.sum() == 0:
    print("✓ No hay valores nulos detectados")
else:
    null_df = pd.DataFrame({'Columna': nulls[nulls > 0].index, 'Nulos': nulls[nulls > 0].values})
    print(null_df.to_string(index=False))

# 2. ESTADÍSTICAS DESCRIPTIVAS GENERALES
print("\n" + "=" * 80)
print("2. ESTADÍSTICAS DESCRIPTIVAS GENERALES")
print("=" * 80)

print("\n[+] Variables numéricas (resumen):")
desc_df = df.describe().round(2)  # Redondeo para claridad
print(desc_df.to_string(float_format='%.0f'))  # Headers ya claros: count, mean (Media), std, etc.

print("\n[+] Variables categóricas (distribución):")
print("\n  • Conteo por 'subtema' (Frecuencia de registros):")
subtema_vc = df['subtema'].value_counts().to_frame('Frecuencia')
print(subtema_vc.to_string())

print("\n  • Top 10 por 'sector' (Frecuencia de registros):")
sector_vc = df['sector'].value_counts().head(10).to_frame('Frecuencia')
print(sector_vc.to_string())

# 3. IDENTIFICACIÓN DE ENTIDADES Y RELACIONES
print("\n" + "=" * 80)
print("3. IDENTIFICACIÓN DE ENTIDADES Y RELACIONES")
print("=" * 80)

print("\n[+] Entidades principales:")
print("  • Tiempo: ciclo + mes (períodos mensuales)")
print("  • Concepto: clave_de_concepto, nombre, subtema (tipos de operaciones fiscales)")
print("  • Sector: sector, ambito (instituciones gubernamentales)")
print("  • Medición: monto + metadatos (valores y unidades)")

print("\nRelaciones clave:")
print("  • Una Medición une Concepto, Sector y Tiempo")
print("  • Un Concepto/Sector tiene múltiples Mediciones temporales")

# Diagrama ER en Mermaid
er_diagram = """
erDiagram
    TIEMPO ||--o{ MEDICION : "ocurre en"
    CONCEPTO ||--o{ MEDICION : "se mide como"
    SECTOR ||--o{ MEDICION : "pertenece a"
    TIEMPO {
        int ciclo PK
        string mes PK
    }
    CONCEPTO {
        string clave_de_concepto PK
        string nombre
        string tema
        string subtema
    }
    SECTOR {
        string sector PK
        string ambito
    }
    MEDICION {
        float monto
        string tipo_de_informacion
        string base_de_registro
        string unidad_de_medida
    }
"""
print("\n[+] Diagrama ER (Mermaid – copia a mermaid.live para visualizar):")
print("-" * 50)
print(er_diagram)

# Guardar en MD
with open('er_diagram.md', 'w') as f:
    f.write("# Diagrama ER: Finanzas Públicas SHCP\n\n```mermaid\n" + er_diagram + "\n```\n")
print("\n✓ Guardado en 'er_diagram.md' (ábrelo en VS Code o GitHub o algún visor de MarkDown para renderizar)")

# 4. ESTADÍSTICAS AGRUPADAS POR ENTIDADES
print("\n" + "=" * 80)
print("4. ESTADÍSTICAS AGRUPADAS POR ENTIDADES")
print("=" * 80)

print("\n[+] Por 'subtema' (estadísticas de monto por categoría fiscal):")
grouped_subtema = df.groupby('subtema')['monto'].agg(['sum', 'mean', 'std', 'min', 'max', 'count'])
grouped_subtema.columns = ['Suma_Total (Miles Pesos)', 'Promedio (Miles Pesos)', 'Desv_Est (Miles Pesos)', 
                           'Minimo (Miles Pesos)', 'Maximo (Miles Pesos)', 'Registros']
print(grouped_subtema.to_string(float_format='%.0f'))

print("\n[+] Por 'sector' (top 10 por suma total de monto):")
grouped_sector = df.groupby('sector')['monto'].agg(['sum', 'mean', 'std'])
grouped_sector.columns = ['Suma_Total (Miles Pesos)', 'Promedio (Miles Pesos)', 'Desv_Est (Miles Pesos)']
print(grouped_sector.sort_values('Suma_Total (Miles Pesos)', ascending=False).head(10).to_string(float_format='%.0f'))

print("\n[+] Por 'ciclo' (tendencias anuales de monto):")
grouped_ciclo = df.groupby('ciclo')['monto'].agg(['sum', 'mean'])
grouped_ciclo.columns = ['Suma_Total (Miles Pesos)', 'Promedio (Miles Pesos)']
print(grouped_ciclo.to_string(float_format='%.0f'))

# 5. EXPORTACIÓN DE VALORES ÚNICOS
print("\n" + "=" * 80)
print("5. EXPORTACIÓN DE VALORES ÚNICOS")
print("=" * 80)

columns_of_interest = ['clave_de_concepto', 'nombre', 'tema', 'subtema', 'sector', 'ambito', 'tipo_de_informacion', 'base_de_registro', 'frecuencia']

print("[+] Generando CSVs (valores únicos + conteos, ordenados alfabéticamente)...")
for col in columns_of_interest:
    vc = df[col].value_counts().sort_index().to_frame('Frecuencia')  # Header en CSV y preview
    csv_filename = f'{col}_unicos.csv'
    vc.to_csv(csv_filename, index_label='Valor_Unico', header=['Frecuencia'])  # Header claro en CSV
    print(f"  [*] '{col}' → {len(vc):,} únicos | Archivo: {csv_filename} (con header 'Frecuencia')")

print("\n[+] ¡Exportación completada! Abre en Excel para explorar categorías.")
print("\n" + "=" * 80)
print("ANÁLISIS FINALIZADO")
print("=" * 80)