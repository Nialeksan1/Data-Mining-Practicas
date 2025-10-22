import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf

# =========================
# CONFIGURACIÓN GENERAL
# =========================
FILE_PATH = "../Ingreso_Gasto_Financiamiento_Pena_AMLO_2012-2024.csv"
OUT_DIR   = "./salidas_practica5"
os.makedirs(OUT_DIR, exist_ok=True)

CUTOFF_DATE      = pd.to_datetime("2018-12-01")  # frontera de sexenios
DEFAULT_SPLIT_DT = pd.to_datetime("2022-01-01")  # split temporal por defecto

# Mostrar números “normales” (no científicos)
pd.options.display.float_format = "{:,.3f}".format
np.set_printoptions(suppress=True)

# Estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("talk")


# =========================
# UTILIDADES
# =========================
def ensure_mes_num(df: pd.DataFrame) -> pd.DataFrame:
    mes_map = {'Enero':1,'Febrero':2,'Marzo':3,'Abril':4,'Mayo':5,'Junio':6,
               'Julio':7,'Agosto':8,'Septiembre':9,'Octubre':10,'Noviembre':11,'Diciembre':12}
    if 'mes_num' not in df.columns:
        df['mes_num'] = df['mes'].map(mes_map)
    return df

def ensure_fecha(df: pd.DataFrame) -> pd.DataFrame:
    # Construir YYYY-MM-01 y normalizar a inicio de mes (sin 'MS' en to_timestamp)
    df['fecha'] = pd.to_datetime(
        df['ciclo'].astype(str) + '-' + df['mes_num'].astype(str) + '-01',
        errors='coerce'
    )
    df['fecha'] = df['fecha'].dt.to_period('M').dt.to_timestamp()  # primer día del mes
    return df.dropna(subset=['fecha'])

def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df['monto'] = pd.to_numeric(df['monto'], errors='coerce')
    return df.dropna(subset=['monto'])

def asinh_series(s: pd.Series) -> pd.Series:
    return np.arcsinh(s.astype(float))

def r2_score_np(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {path}")

def temporal_split(df: pd.DataFrame, split_date: pd.Timestamp):
    train = df[df['fecha'] < split_date].copy()
    test  = df[df['fecha'] >= split_date].copy()
    if train.empty or test.empty:
        # Fallback: 80/20 por tiempo si no hay datos a ambos lados
        cutoff = df['fecha'].sort_values().iloc[int(0.8*len(df))]
        train = df[df['fecha'] < cutoff].copy()
        test  = df[df['fecha'] >= cutoff].copy()
        print(f"⚠️ Split por defecto no viable; usando fallback 80/20 con corte en {cutoff.date()}.")
    return train, test

def get_term_names(rob, ols=None):
    """Nombres de variables robusto a versiones de statsmodels."""
    if hasattr(rob, "model") and hasattr(rob.model, "exog_names"):
        return list(rob.model.exog_names)
    if ols is not None and hasattr(ols.params, "index"):
        return list(ols.params.index)
    return [f"x{i}" for i in range(len(np.atleast_1d(rob.params)))]


# =========================
# 1) CARGA Y PREPARACIÓN
# =========================
print(f"Cargando dataset: {FILE_PATH}")
df = pd.read_csv(FILE_PATH, low_memory=False)
print(f"✓ Datos: {df.shape[0]:,} filas × {df.shape[1]} columnas")

# Validaciones mínimas
for col in ['ciclo','mes','monto','subtema']:
    if col not in df.columns:
        raise ValueError(f"Falta columna requerida: {col}")

df = ensure_mes_num(df)
df = ensure_fecha(df)
df = ensure_numeric(df)

# Etiqueta de sexenio
df['sexenio'] = np.where(df['fecha'] < CUTOFF_DATE, 'Peña Nieto', 'AMLO')

# Panel mensual por subtema (sumas mensuales comparables)
monthly = (df.groupby(['fecha','sexenio','subtema'], observed=True, as_index=False)['monto']
             .sum()
             .sort_values('fecha'))
monthly['monto_asinh'] = asinh_series(monthly['monto'])

# Variables de tendencia y estacionalidad
monthly['mes'] = monthly['fecha'].dt.month
monthly['sin12'] = np.sin(2*np.pi*monthly['mes']/12.0)
monthly['cos12'] = np.cos(2*np.pi*monthly['mes']/12.0)

# Índice temporal lineal (meses desde el inicio)
t_index = monthly['fecha'].dt.year*12 + monthly['fecha'].dt.month
monthly['t'] = t_index - t_index.min()

# Rezago dentro de cada subtema (persistencia)
monthly['lag1'] = (monthly.sort_values('fecha')
                          .groupby('subtema', observed=True)['monto_asinh']
                          .shift(1))

# Binarización de sexenio (solo para correlación)
monthly['sexenio_bin'] = (monthly['sexenio'] == 'AMLO').astype(int)

# Conjunto final para el modelo
model_df = monthly.dropna(subset=['lag1']).copy()

print("\nMuestra del panel listo para modelar:")
print(model_df.head(8))


# =========================
# 2) CORRELACIONES (HEATMAPS)
# =========================
print("\nCalculando correlaciones (Pearson y Spearman) y generando heatmaps...")
corr_cols = ['monto_asinh','t','sin12','cos12','lag1','sexenio_bin']
pear = model_df[corr_cols].corr(method='pearson')
spear = model_df[corr_cols].corr(method='spearman')

for name, M in [('pearson', pear), ('spearman', spear)]:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(M, annot=True, fmt=".2f", cmap='coolwarm', center=0,
                linewidths=0.5, linecolor='white', ax=ax)
    ax.set_title(f"Correlación ({name.title()}) – variables del modelo")
    fig.tight_layout()
    savefig(fig, f"corr_heatmap_{name}.png")
    plt.close(fig)


# =========================
# 3) MODELO LINEAL (OLS con robustez HC3)
# =========================
# Fórmula: target ~ tendencia + estacionalidad + rezago + sexenio + subtema + interacción
formula = "monto_asinh ~ t + sin12 + cos12 + lag1 + C(sexenio) * C(subtema)"

# Split temporal (con fallback si es necesario)
train, test = temporal_split(model_df, DEFAULT_SPLIT_DT)
print(f"\nSplit temporal -> Train: {train.shape[0]:,} filas | Test: {test.shape[0]:,} filas")

# En casos raros, asegurar que no haya niveles de subtema en test que no existan en train
if not set(test['subtema']).issubset(set(train['subtema'])):
    missing = sorted(set(test['subtema']) - set(train['subtema']))
    print(f"⚠️ Subtemas en test no vistos en train: {missing}. Se excluirán del test.")
    test = test[test['subtema'].isin(train['subtema'])].copy()

# Ajuste
ols = smf.ols(formula, data=train).fit()
# Covarianza robusta (HC3) para errores estándar/p-valores
rob = ols.get_robustcov_results(cov_type='HC3')

print("\nResumen OLS (robusto HC3):")
print(rob.summary())

# Métricas R²
r2_in  = float(ols.rsquared)
r2_adj = float(ols.rsquared_adj)

# Predicción out-of-sample (usar ols.predict; coeficientes idénticos)
yhat_test = ols.predict(test)
r2_out = r2_score_np(test['monto_asinh'].values, yhat_test.values)

print(f"\nR² (train, in-sample): {r2_in:.3f}  |  R² ajustado: {r2_adj:.3f}")
print(f"R² (test, out-of-sample): {r2_out:.3f}")

# =========================
# 4) TABLA DE COEFICIENTES
# =========================
term_names = get_term_names(rob, ols)

coef_tbl = pd.DataFrame({
    'term'   : term_names,
    'coef'   : np.atleast_1d(rob.params),
    'std_err': np.atleast_1d(rob.bse),
    't_value': np.atleast_1d(rob.tvalues),
    'p_value': np.atleast_1d(rob.pvalues),
})
coef_tbl['ci95_low']  = coef_tbl['coef'] - 1.96 * coef_tbl['std_err']
coef_tbl['ci95_high'] = coef_tbl['coef'] + 1.96 * coef_tbl['std_err']
coef_tbl.to_csv(os.path.join(OUT_DIR, "coeficientes_ols_hc3.csv"), index=False)

# Predicciones en test
pred_test = test[['fecha','sexenio','subtema','monto_asinh']].copy()
pred_test['y_pred'] = yhat_test.values
pred_test.to_csv(os.path.join(OUT_DIR, "predicciones_test.csv"), index=False)


# =========================
# 5) GRÁFICAS DEL MODELO
# =========================
sns.set_context("talk")

# 5.1 Predicho vs Real
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(pred_test['monto_asinh'], pred_test['y_pred'], alpha=0.6, s=28)
xlim = ax.get_xlim(); ylim = ax.get_ylim()
lims = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
ax.plot(lims, lims, '--', color='gray', linewidth=1)  # línea 45°
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_title(f"Predicho vs Real (Test) – asinh(monto)\nR²_test = {r2_out:.3f}")
ax.set_xlabel("Real"); ax.set_ylabel("Predicho")
fig.tight_layout()
savefig(fig, "pred_vs_real_test.png")
plt.close(fig)

# 5.2 Residuales vs Ajustados
resid = pred_test['monto_asinh'] - pred_test['y_pred']
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(pred_test['y_pred'], resid, alpha=0.6, s=24)
ax.axhline(0, color='gray', linestyle='--')
ax.set_title("Residuales vs Ajustados (Test)")
ax.set_xlabel("Ajustados (predicho)"); ax.set_ylabel("Residuales")
fig.tight_layout()
savefig(fig, "resid_vs_fitted_test.png")
plt.close(fig)

# 5.3 Líneas: Real vs Predicho por subtema
for subt, g in pred_test.groupby('subtema'):
    g2 = g.sort_values('fecha').copy()
    if g2.empty:
        continue
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(g2['fecha'], g2['monto_asinh'], marker='o', linewidth=1.5, label='Real')
    ax.plot(g2['fecha'], g2['y_pred'], marker='.', linewidth=1.5, label='Predicho')
    ax.axvline(CUTOFF_DATE, ls='--', alpha=0.8, label='Cambio sexenio')
    ax.set_title(f"asinh(monto) – Real vs Predicho (Test) | Subtema: {subt}")
    ax.set_xlabel("Fecha"); ax.set_ylabel("asinh(monto)")
    ax.legend()
    fig.tight_layout()
    fn = f"line_real_vs_pred_{subt.replace(' ','_')}_test.png"
    savefig(fig, fn)
    plt.close(fig)

# 5.4 Coeficientes (barra con IC95%)
fig, ax = plt.subplots(figsize=(10, max(5, 0.35*len(coef_tbl))))
plot_tbl = coef_tbl.sort_values('coef')
ax.barh(plot_tbl['term'], plot_tbl['coef'], xerr=1.96*plot_tbl['std_err'], alpha=0.8)
ax.axvline(0, color='k', linewidth=1)
ax.set_title("Coeficientes OLS (errores robustos HC3) – IC95%")
ax.set_xlabel("Estimación (escala asinh)")
fig.tight_layout()
savefig(fig, "coeficientes_bar_ic.png")
plt.close(fig)


print("\n✓ Archivos guardados en:", os.path.abspath(OUT_DIR))
print("   - corr_heatmap_pearson.png / corr_heatmap_spearman.png")
print("   - pred_vs_real_test.png, resid_vs_fitted_test.png")
print("   - line_real_vs_pred_<subtema>_test.png")
print("   - coeficientes_bar_ic.png, coeficientes_ols_hc3.csv")
print(f"\nR² (train): {r2_in:.3f} | R² adj.: {r2_adj:.3f} | R² (test): {r2_out:.3f}")
