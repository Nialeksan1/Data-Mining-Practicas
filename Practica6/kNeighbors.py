import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Formato de impresión “normal”
pd.options.display.float_format = "{:,.3f}".format
np.set_printoptions(suppress=True)

# --------------------------
# Intentar importar sklearn
# --------------------------
try:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        classification_report, confusion_matrix, roc_auc_score, roc_curve
    )
except Exception as e:
    raise SystemExit(
        "No se pudo importar scikit-learn. Instala con:\n"
        "  pip install -U scikit-learn\n\n"
        f"Detalle: {e}"
    )

# =========================
# CONFIGURACIÓN GENERAL
# =========================
FILE_PATH = "../Ingreso_Gasto_Financiamiento_Pena_AMLO_2012-2024.csv"
OUT_DIR   = "./salidas_practica6"
os.makedirs(OUT_DIR, exist_ok=True)

CUTOFF_DATE = pd.to_datetime("2018-12-01")          # Frontera de sexenios
SPLIT_DATE  = pd.to_datetime("2022-01-01")          # Corte temporal para test
RANDOM_SEED = 42

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
    df['fecha'] = df['fecha'].dt.to_period('M').dt.to_timestamp()  # primer día de mes
    return df.dropna(subset=['fecha'])

def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df['monto'] = pd.to_numeric(df['monto'], errors='coerce')
    return df.dropna(subset=['monto'])

def asinh_series(s: pd.Series) -> pd.Series:
    return np.arcsinh(s.astype(float))

def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {path}")

def make_ohe():
    # Compatibilidad con versiones de scikit-learn
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

# =========================
# 1) CARGA Y PREPARACIÓN
# =========================
print(f"Cargando dataset: {FILE_PATH}")
df = pd.read_csv(FILE_PATH, low_memory=False)
print(f"✓ Datos: {df.shape[0]:,} filas × {df.shape[1]} columnas")

# Validaciones mínimas
for c in ['ciclo','mes','monto','subtema','sector']:
    if c not in df.columns:
        raise ValueError(f"Falta columna requerida: {c}")

df = ensure_mes_num(df)
df = ensure_fecha(df)
df = ensure_numeric(df)

# Etiqueta objetivo: sexenio
df['sexenio'] = np.where(df['fecha'] < CUTOFF_DATE, 'Peña Nieto', 'AMLO')

# =========================
# 2) PANEL MENSUAL + FEATURES
# =========================
# Agregamos por (fecha, subtema, sector) para tener observaciones comparables
monthly = (
    df.groupby(['fecha','subtema','sector'], observed=True, as_index=False)['monto']
      .sum()
      .sort_values('fecha')
)
monthly['sexenio'] = np.where(monthly['fecha'] < CUTOFF_DATE, 'Peña Nieto', 'AMLO')
monthly['monto_asinh'] = asinh_series(monthly['monto'])

# Estacionalidad (mes del año en sin/cos)
monthly['mes']   = monthly['fecha'].dt.month
monthly['sin12'] = np.sin(2*np.pi*monthly['mes']/12.0)
monthly['cos12'] = np.cos(2*np.pi*monthly['mes']/12.0)

# Rezagos y rolling dentro de cada (subtema, sector)
group_cols = ['subtema','sector']
monthly = monthly.sort_values(['subtema','sector','fecha']).copy()

# lag1 del monto_asinh
monthly['lag1'] = monthly.groupby(group_cols, observed=True)['monto_asinh'].shift(1)
# rolling mean/std (ventana 3 meses)
monthly['roll3_mean'] = monthly.groupby(group_cols, observed=True)['monto_asinh'].rolling(3).mean().reset_index(level=group_cols, drop=True)
monthly['roll3_std']  = monthly.groupby(group_cols, observed=True)['monto_asinh'].rolling(3).std().reset_index(level=group_cols, drop=True)

# Eliminamos filas con NaN generados por lag/rolling (para entrenamiento limpio)
model_df = monthly.dropna(subset=['lag1','roll3_mean','roll3_std']).copy()

# Features y target
num_features = ['monto_asinh','lag1','roll3_mean','roll3_std','sin12','cos12']
cat_features = ['subtema','sector']
target_col   = 'sexenio'

print("\nMuestra de features:")
print(model_df[[*num_features, *cat_features, target_col]].head(8))

# =========================
# 3) SPLIT HOLD-OUT (TEMPORAL) + CV
# =========================
# Split temporal para evaluación final
train_df = model_df[model_df['fecha'] < SPLIT_DATE].copy()
test_df  = model_df[model_df['fecha'] >= SPLIT_DATE].copy()

if train_df.empty or test_df.empty:
    # Fallback 80/20 temporal si la fecha elegida no separa bien
    cutoff = model_df['fecha'].sort_values().iloc[int(0.8*len(model_df))]
    train_df = model_df[model_df['fecha'] < cutoff].copy()
    test_df  = model_df[model_df['fecha'] >= cutoff].copy()
    print(f"⚠️ Ajuste de split: usando 80/20 con corte en {cutoff.date()}.")

print(f"\nSplit temporal -> Train: {train_df.shape[0]:,} filas | Test: {test_df.shape[0]:,} filas")

X_train = train_df[ num_features + cat_features ]
y_train = train_df[ target_col ]
X_test  = test_df[  num_features + cat_features ]
y_test  = test_df[  target_col ]

# Para ROC-AUC binario, definimos 1 = AMLO
y_train_bin = (y_train == 'AMLO').astype(int)
y_test_bin  = (y_test  == 'AMLO').astype(int)

# =========================
# 4) PIPELINE + GRID SEARCH
# =========================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     make_ohe())
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ],
    remainder="drop"
)

knn = KNeighborsClassifier()

pipe = Pipeline(steps=[
    ("prep", preprocess),
    ("clf",  knn)
])

# Validación cruzada temporal (mantiene orden cronológico)
tscv = TimeSeriesSplit(n_splits=5)

param_grid = {
    "clf__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31, 41],
    "clf__weights":     ["uniform", "distance"],
    "clf__p":           [1, 2],  # Manhattan vs Euclidiana
}

print("\nGridSearchCV (TimeSeriesSplit=5) en curso...")
gscv = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="roc_auc",           # AUC binario (usa y_bin internamente)
    cv=tscv,
    n_jobs=-1,
    verbose=1
)
gscv.fit(X_train, y_train_bin)   # y binario para ROC-AUC

print("\nMejores hiper-parámetros (CV):")
print(gscv.best_params_)
print(f"Mejor AUC (CV): {gscv.best_score_:.3f}")

best_model = gscv.best_estimator_

# =========================
# 5) EVALUACIÓN HOLD-OUT
# =========================
# --- PREDICCIÓN ---
# Predicción del modelo entrenado con binarios => devuelve 0/1
y_pred_bin = best_model.predict(X_test)

# Prob de clase positiva (AMLO). Resolver índice de la clase positiva de forma robusta.
def get_pos_index_for_amlo(model):
    # 1) Pipeline puede tener .classes_
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    # 2) O en el estimador final 'clf'
    elif hasattr(model, "named_steps") and 'clf' in model.named_steps and hasattr(model.named_steps['clf'], "classes_"):
        classes = list(model.named_steps['clf'].classes_)
    else:
        return None
    # Preferimos 1; si no existe, intentar 'AMLO' o True
    if 1 in classes: return classes.index(1)
    if 'AMLO' in classes: return classes.index('AMLO')
    if True in classes: return classes.index(True)
    # fallback: última clase
    return len(classes) - 1

pos_idx = get_pos_index_for_amlo(best_model)
if pos_idx is not None and hasattr(best_model, "predict_proba"):
    y_proba = best_model.predict_proba(X_test)[:, pos_idx]
else:
    y_proba = None

# --- ARREGLO CLAVE: TIPOS CONSISTENTES PARA MÉTRICAS ---
# Mapear 0/1 -> strings para comparar contra y_test (strings)
LABEL_MAP_BIN2STR = {0: 'Peña Nieto', 1: 'AMLO'}
if np.issubdtype(np.asarray(y_pred_bin).dtype, np.number):
    y_pred = pd.Series(y_pred_bin).map(LABEL_MAP_BIN2STR).values
else:
    # Si por algún motivo el pipeline devolviera strings, no mapear
    y_pred = y_pred_bin

# --- MÉTRICAS EN STRINGS (y_test vs y_pred) ---
acc  = accuracy_score(y_test, y_pred)
bacc = balanced_accuracy_score(y_test, y_pred)
f1_p = f1_score(y_test, y_pred, pos_label='Peña Nieto')
f1_a = f1_score(y_test, y_pred, pos_label='AMLO')

print(f"\nHold-out test | Accuracy: {acc:.3f} | Balanced Acc: {bacc:.3f} | F1(AMLO): {f1_a:.3f} | F1(Peña): {f1_p:.3f}")
print("\nClassification report (test):")
print(classification_report(y_test, y_pred, digits=3))

# ROC-AUC en binario (AMLO positivo)
if y_proba is not None:
    auc = roc_auc_score(y_test_bin, y_proba)
    print(f"ROC-AUC (test): {auc:.3f}")
else:
    auc = np.nan

# =========================
# 6) GRÁFICAS: MATRIZ CONFUSIÓN + ROC
# =========================
# Matriz de confusión (con etiquetas de texto)
cm = confusion_matrix(y_test, y_pred, labels=['Peña Nieto','AMLO'])
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=['Peña Nieto','AMLO'],
            yticklabels=['Peña Nieto','AMLO'],
            ax=ax)
ax.set_title("Matriz de Confusión (Test)")
ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
fig.tight_layout()
savefig(fig, "confusion_matrix_test.png")
plt.close(fig)

# Curva ROC
if y_proba is not None:
    fpr, tpr, thr = roc_curve(y_test_bin, y_proba)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0,1],[0,1],'--', color='gray')
    ax.set_title("Curva ROC (Clase positiva: AMLO)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc='lower right')
    fig.tight_layout()
    savefig(fig, "roc_curve_test.png")
    plt.close(fig)

# =========================
# 7) EXPORTACIÓN DE RESULTADOS
# =========================
# Predicciones detalladas
pred_out = test_df[['fecha','subtema','sector','sexenio','monto','monto_asinh','lag1','roll3_mean','roll3_std','sin12','cos12']].copy()
pred_out['pred_sexenio'] = y_pred
pred_out['is_correct']   = (pred_out['sexenio'] == pred_out['pred_sexenio']).astype(int)
if y_proba is not None:
    pred_out['proba_AMLO'] = y_proba

pred_csv = os.path.join(OUT_DIR, "predicciones_knn_test.csv")
pred_out.to_csv(pred_csv, index=False)
print(f"\n✓ Predicciones guardadas en: {pred_csv}")

print("\n✓ Archivos guardados en:", os.path.abspath(OUT_DIR))
print("   - confusion_matrix_test.png")
print("   - roc_curve_test.png")
print("   - predicciones_knn_test.csv")
