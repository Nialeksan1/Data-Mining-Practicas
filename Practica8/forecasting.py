import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# --------------------------
# Configuración general
# --------------------------
FILE_PATH = "../Ingreso_Gasto_Financiamiento_Pena_AMLO_2012-2024.csv"
OUT_DIR   = "./salidas_practica8"
os.makedirs(OUT_DIR, exist_ok=True)

CUTOFF_DATE   = pd.to_datetime("2018-12-01")     # solo para referencia, no se usa como feature
TEST_START    = pd.to_datetime("2022-01-01")     # inicio del hold-out temporal
FORECAST_H    = 12                               # meses a futuro
RANDOM_SEED   = 42

# Formato “normal”
pd.options.display.float_format = "{:,.3f}".format
np.set_printoptions(suppress=True)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("talk")

# --------------------------
# Utilidades
# --------------------------
def ensure_mes_num(df: pd.DataFrame) -> pd.DataFrame:
    mes_map = {'Enero':1,'Febrero':2,'Marzo':3,'Abril':4,'Mayo':5,'Junio':6,
               'Julio':7,'Agosto':8,'Septiembre':9,'Octubre':10,'Noviembre':11,'Diciembre':12}
    if 'mes_num' not in df.columns:
        df['mes_num'] = df['mes'].map(mes_map)
    return df

def ensure_fecha(df: pd.DataFrame) -> pd.DataFrame:
    df['fecha'] = pd.to_datetime(
        df['ciclo'].astype(str) + '-' + df['mes_num'].astype(str) + '-01',
        errors='coerce'
    )
    # Normalizar a inicio de mes (sin 'MS' en to_timestamp)
    df['fecha'] = df['fecha'].dt.to_period('M').dt.to_timestamp()
    return df.dropna(subset=['fecha'])

def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df['monto'] = pd.to_numeric(df['monto'], errors='coerce')
    return df.dropna(subset=['monto'])

def asinh(x):
    return np.arcsinh(np.asarray(x, float))

def sinv(x):  # inversa de asinh
    return np.sinh(np.asarray(x, float))

def month_sin_cos(month_idx):
    return np.sin(2*np.pi*month_idx/12.0), np.cos(2*np.pi*month_idx/12.0)

def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {path}")

def build_feature_row(dt, y_hist_asinh, t0, dates_index):
    """
    Crea un vector de features para la fecha dt usando:
      t     : índice temporal lineal (meses desde inicio)
      sin12 : estacionalidad seno
      cos12 : estacionalidad coseno
      lag1  : y_{t-1} (asinh) tomado de y_hist_asinh
      lag12 : y_{t-12} (asinh) tomado de y_hist_asinh
    y_hist_asinh debe contener valores (observados o predichos) por fecha.
    """
    # índice temporal
    t = (dt.year*12 + dt.month) - t0
    s, c = month_sin_cos(dt.month)

    # lags
    prev_1m = (dt + pd.offsets.DateOffset(months=-1)).to_period('M').to_timestamp()
    prev_12m = (dt + pd.offsets.DateOffset(months=-12)).to_period('M').to_timestamp()

    lag1  = y_hist_asinh.get(prev_1m, np.nan)
    lag12 = y_hist_asinh.get(prev_12m, np.nan)

    return pd.DataFrame(
        {"t":[t], "sin12":[s], "cos12":[c], "lag1":[lag1], "lag12":[lag12]},
        index=[dt]
    )

def dynamic_predict(pipe, dates, y_hist_asinh, t0):
    """
    Predicción dinámica sobre una secuencia de fechas 'dates' (ordenadas):
    usa y_hist_asinh (observado/predicho) para construir lags sin fugas de info.
    Retorna un DataFrame con columnas yhat_asinh y yhat (monto).
    """
    rows = []
    for dt in dates:
        X_row = build_feature_row(dt, y_hist_asinh, t0, dates)
        if X_row[['lag1','lag12']].isna().any(axis=None):
            # Si falta lag (al inicio de la serie), saltar
            rows.append((dt, np.nan, np.nan))
            continue
        yhat_asinh = float(pipe.predict(X_row)[0])
        y_hist_asinh[dt] = yhat_asinh  # alimentar el siguiente paso
        rows.append((dt, yhat_asinh, float(sinv(yhat_asinh))))
    return pd.DataFrame(rows, columns=['fecha','yhat_asinh','yhat']).set_index('fecha')

# --------------------------
# 1) Carga y preparación
# --------------------------
print(f"Cargando dataset: {FILE_PATH}")
df = pd.read_csv(FILE_PATH, low_memory=False)
print(f"✓ Datos: {df.shape[0]:,} filas × {df.shape[1]} columnas")

for col in ['ciclo','mes','monto']:
    if col not in df.columns:
        raise ValueError(f"Falta columna requerida: {col}")

df = ensure_mes_num(df)
df = ensure_fecha(df)
df = ensure_numeric(df)

# Serie total mensual
ts = (df.groupby('fecha', as_index=False)['monto'].sum()
        .sort_values('fecha')
        .set_index('fecha'))

# Objetivo transformado
ts['y'] = asinh(ts['monto'])

# Fechas y t0
dates_all = ts.index.to_period('M').to_timestamp()
t0 = (dates_all[0].year*12 + dates_all[0].month)

# --------------------------
# 2) Construcción de features para entrenamiento
# --------------------------
feat_rows = []
for dt in dates_all:
    t = (dt.year*12 + dt.month) - t0
    s, c = month_sin_cos(dt.month)
    prev_1m = (dt + pd.offsets.DateOffset(months=-1)).to_period('M').to_timestamp()
    prev_12m = (dt + pd.offsets.DateOffset(months=-12)).to_period('M').to_timestamp()
    lag1  = ts.loc[prev_1m, 'y'] if prev_1m in ts.index else np.nan
    lag12 = ts.loc[prev_12m, 'y'] if prev_12m in ts.index else np.nan
    feat_rows.append((dt, t, s, c, lag1, lag12))

Xall = (pd.DataFrame(feat_rows, columns=['fecha','t','sin12','cos12','lag1','lag12'])
          .set_index('fecha'))
yall = ts['y'].copy()

# Remover filas sin lags (primeros 12 meses)
valid = ~Xall[['lag1','lag12']].isna().any(axis=1)
Xall = Xall[valid]; yall = yall.loc[Xall.index]

# Split temporal
train_idx = Xall.index < TEST_START
Xtrain, ytrain = Xall[train_idx], yall[train_idx]
Xtest,  ytest  = Xall[~train_idx], yall[~train_idx]

print(f"Split temporal -> Train: {Xtrain.shape[0]:,} filas | Test: {Xtest.shape[0]:,} filas "
      f"(test desde {Xtest.index.min().date() if not Xtest.empty else 'N/A'})")

# --------------------------
# 3) Modelo lineal (pipeline)
# --------------------------
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr",     LinearRegression())
])
pipe.fit(Xtrain, ytrain)

# --------------------------
# 4) Predicción dinámica en TEST (sin ver futuro)
# --------------------------
# Historial hasta el último punto de train (para lags)
y_hist_asinh = ts.loc[:Xtrain.index.max(), 'y'].to_dict()

# Fechas de test en orden
test_dates = list(Xtest.index)
pred_test = dynamic_predict(pipe, test_dates, y_hist_asinh.copy(), t0)

# Métricas en test (solo donde hay yhat)
eval_idx = pred_test['yhat_asinh'].notna()
y_true_test_asinh = ytest.loc[eval_idx.index[eval_idx]].values if hasattr(eval_idx, 'index') else ytest.values
y_pred_test_asinh = pred_test.loc[eval_idx, 'yhat_asinh'].values

# R²/MAE/RMSE en escala asinh
r2_test_asinh = r2_score(y_true_test_asinh, y_pred_test_asinh)
mae_asinh = mean_absolute_error(y_true_test_asinh, y_pred_test_asinh)
rmse_asinh = float(np.sqrt(np.mean((y_true_test_asinh - y_pred_test_asinh)**2)))

# También en escala original (monto)
y_true_test = sinv(y_true_test_asinh)
y_pred_test = sinv(y_pred_test_asinh)
mae = mean_absolute_error(y_true_test, y_pred_test)
rmse = float(np.sqrt(np.mean((y_true_test - y_pred_test)**2)))

print("\n== Métricas TEST ==")
print(f"R² (asinh): {r2_test_asinh:.3f} | MAE: {mae:,.0f} | RMSE: {rmse:,.0f}")

# --------------------------
# 5) Forecast H pasos hacia delante
# --------------------------
last_date = dates_all.max()
future_dates = pd.period_range(last_date, periods=FORECAST_H+1, freq='M')[1:].to_timestamp()
pred_future = dynamic_predict(pipe, list(future_dates), y_hist_asinh.copy(), t0)

# --------------------------
# 6) Reconstruir serie para exportar y graficar
# --------------------------
out = pd.DataFrame(index=ts.index)
out['y_true_asinh'] = ts['y']
out['y_true'] = ts['monto']

# Ajuste dentro de TRAIN (fitted estático con lags observados)
yhat_train_asinh = pipe.predict(Xtrain)
out.loc[Xtrain.index, 'yhat_asinh'] = yhat_train_asinh
out.loc[Xtrain.index, 'yhat'] = sinv(yhat_train_asinh)

# Predicción dinámica TEST
out.loc[pred_test.index, 'yhat_asinh'] = pred_test['yhat_asinh']
out.loc[pred_test.index, 'yhat'] = pred_test['yhat']

# Forecast futuro
fc = pred_future.copy()
fc['y_true_asinh'] = np.nan
fc['y_true'] = np.nan

# CSV combinado
export = pd.concat([out, fc[['yhat_asinh','yhat','y_true_asinh','y_true']]], axis=0)
export = export.reset_index().rename(columns={'index':'fecha'})
export['split'] = np.where(export['fecha'] < TEST_START, 'train',
                   np.where(export['fecha'] <= last_date, 'test', 'forecast'))
csv_path = os.path.join(OUT_DIR, "forecast_linear_regression.csv")
export.to_csv(csv_path, index=False)
print(f"✓ Guardado CSV: {csv_path}")

# --------------------------
# 7) Gráficas
# --------------------------
# 7.1 Serie completa: Real vs Predicho + Forecast
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(export['fecha'], export['y_true'], label='Real', linewidth=1.8)
ax.plot(export['fecha'], export['yhat'],  label='Predicho/Forecast', linewidth=1.8)
ax.axvline(TEST_START, ls='--', color='gray', alpha=0.8, label='Inicio Test')
ax.axvline(last_date, ls=':', color='gray', alpha=0.8, label='Últ. dato real')
ax.set_title(f"Forecast con Regresión Lineal (H={FORECAST_H})\nR²_test(asinh)={r2_test_asinh:.3f} | MAE={mae:,.0f} | RMSE={rmse:,.0f}")
ax.set_xlabel("Fecha")
ax.set_ylabel("Monto (Miles de Pesos)")
ax.legend()
fig.tight_layout()
savefig(fig, "forecast_lineal_serie.png")
plt.close(fig)

# 7.2 Residuales en TEST (escala original)
res_test = pd.Series(y_true_test - y_pred_test, index=pred_test.loc[eval_idx, :].index)
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(res_test.index, res_test.values, marker='.', linewidth=1.2)
ax.axhline(0, color='gray', ls='--')
ax.set_title("Residuales en Test (monto)")
ax.set_xlabel("Fecha")
ax.set_ylabel("Error (Real - Predicho)")
fig.tight_layout()
savefig(fig, "forecast_residuales_test.png")
plt.close(fig)

# 7.3 Scatter Real vs Predicho en TEST (escala asinh)
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y_true_test_asinh, y_pred_test_asinh, alpha=0.6, s=30)
lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, '--', color='gray')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_title("Real vs Predicho (TEST) – asinh(monto)")
ax.set_xlabel("Real"); ax.set_ylabel("Predicho")
fig.tight_layout()
savefig(fig, "forecast_scatter_test_asinh.png")
plt.close(fig)

print("\n✓ Listo. Revisa la carpeta:", os.path.abspath(OUT_DIR))
print("   - forecast_linear_regression.csv")
print("   - forecast_lineal_serie.png")
print("   - forecast_residuales_test.png")
print("   - forecast_scatter_test_asinh.png")
