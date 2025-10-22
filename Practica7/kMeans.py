import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, adjusted_mutual_info_score, v_measure_score
)

# ---------- Configuración ----------
FILE_PATH = "../Ingreso_Gasto_Financiamiento_Pena_AMLO_2012-2024.csv"
OUT_DIR   = "./salidas_practica7"
os.makedirs(OUT_DIR, exist_ok=True)

CUTOFF_DATE  = pd.to_datetime("2018-12-01")   # frontera sexenio (solo para etiqueta externa)
SPLIT_DATE   = pd.to_datetime("2022-01-01")   # hold-out temporal
RANDOM_SEED  = 42

pd.options.display.float_format = "{:,.3f}".format
np.set_printoptions(suppress=True)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("talk")

# ---------- Utilidades ----------
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
    df['fecha'] = df['fecha'].dt.to_period('M').dt.to_timestamp()
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
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

def safe_kmeans(n_clusters, random_state=RANDOM_SEED):
    # Compatibilidad n_init según versión de sklearn
    try:
        return KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    except TypeError:
        return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)

# ---------- 1) Carga y preparación ----------
print(f"Cargando dataset: {FILE_PATH}")
df = pd.read_csv(FILE_PATH, low_memory=False)
print(f"✓ Datos: {df.shape[0]:,} filas × {df.shape[1]} columnas")

for c in ['ciclo','mes','monto','subtema','sector']:
    if c not in df.columns:
        raise ValueError(f"Falta columna requerida: {c}")

df = ensure_mes_num(df)
df = ensure_fecha(df)
df = ensure_numeric(df)

# Etiqueta externa (NO usar como feature): sexenio
df['sexenio'] = np.where(df['fecha'] < CUTOFF_DATE, 'Peña Nieto', 'AMLO')

# Panel mensual por (subtema, sector)
monthly = (
    df.groupby(['fecha','subtema','sector'], observed=True, as_index=False)['monto']
      .sum()
      .sort_values('fecha')
)
monthly['sexenio']     = np.where(monthly['fecha'] < CUTOFF_DATE, 'Peña Nieto', 'AMLO')  # solo para evaluación externa
monthly['monto_asinh'] = asinh_series(monthly['monto'])
monthly['mes']         = monthly['fecha'].dt.month
monthly['sin12']       = np.sin(2*np.pi*monthly['mes']/12.0)
monthly['cos12']       = np.cos(2*np.pi*monthly['mes']/12.0)

# Rezagos y rolling por (subtema, sector)
monthly = monthly.sort_values(['subtema','sector','fecha']).copy()
group_cols = ['subtema','sector']
monthly['lag1']        = monthly.groupby(group_cols, observed=True)['monto_asinh'].shift(1)
monthly['roll3_mean']  = monthly.groupby(group_cols, observed=True)['monto_asinh'].rolling(3).mean().reset_index(level=group_cols, drop=True)
monthly['roll3_std']   = monthly.groupby(group_cols, observed=True)['monto_asinh'].rolling(3).std().reset_index(level=group_cols, drop=True)

# Limpieza de NaN por ingeniería
model_df = monthly.dropna(subset=['lag1','roll3_mean','roll3_std']).copy()

# ---------- 2) Features ----------
num_features = ['monto_asinh','lag1','roll3_mean','roll3_std','sin12','cos12']
cat_features = ['subtema','sector']  # categóricas, no incluimos sexenio

X_all  = model_df[num_features + cat_features]
y_ext  = model_df['sexenio']  # solo para evaluación externa
fechas = model_df['fecha']    # para split y visualización

# Split temporal
train_mask = fechas < SPLIT_DATE
test_mask  = ~train_mask

X_train, X_test = X_all[train_mask], X_all[test_mask]
y_train_ext     = y_ext[train_mask]   # etiquetas externas (strings)
y_test_ext      = y_ext[test_mask]

print(f"\nSplit temporal -> Train: {X_train.shape[0]:,} filas | Test: {X_test.shape[0]:,} filas")
if X_train.empty or X_test.empty:
    cutoff = fechas.sort_values().iloc[int(0.8*len(fechas))]
    train_mask = fechas < cutoff
    test_mask  = ~train_mask
    X_train, X_test = X_all[train_mask], X_all[test_mask]
    y_train_ext, y_test_ext = y_ext[train_mask], y_ext[test_mask]
    print(f"⚠️ Ajuste de split: usando 80/20 con corte en {cutoff.date()}.")

# Preprocesamiento
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     make_ohe()),
])
prep = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ],
    remainder="drop"
)

# Transformaciones
X_train_prep = prep.fit_transform(X_train)
X_test_prep  = prep.transform(X_test)

# ---------- 3) Selección de K: Elbow + Silhouette ----------
k_list = list(range(2, 11))
inertias, sils = [], []

# Para silhouette costoso en grandes N, muestreamos para score (etiquetas del modelo en full)
def sample_idx(n, max_n=20000, seed=RANDOM_SEED):
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False))

train_sample_idx = sample_idx(X_train_prep.shape[0], max_n=20000)

for k in k_list:
    km = safe_kmeans(n_clusters=k)
    labels_train = km.fit_predict(X_train_prep)
    inertias.append(km.inertia_)

    # Silhouette sobre muestra (usando etiquetas de todo el train)
    s_idx = train_sample_idx
    sil = silhouette_score(X_train_prep[s_idx], labels_train[s_idx])
    sils.append(sil)

# Elegimos K con mejor silhouette (se puede ponderar con elbow si deseas)
best_k = int(k_list[int(np.argmax(sils))])
print(f"\nSelección de K -> mejor silhouette en train: K = {best_k}  (sil={max(sils):.3f})")

# Graficar Elbow y Silhouette vs K
fig, ax = plt.subplots(1, 2, figsize=(12,5))
ax[0].plot(k_list, inertias, marker='o')
ax[0].set_title("Elbow (Inertia) – Train")
ax[0].set_xlabel("K"); ax[0].set_ylabel("Inertia")
ax[0].grid(True, alpha=0.3)

ax[1].plot(k_list, sils, marker='o')
ax[1].axvline(best_k, ls='--', color='gray')
ax[1].set_title("Silhouette vs K – Train")
ax[1].set_xlabel("K"); ax[1].set_ylabel("Silhouette")
ax[1].grid(True, alpha=0.3)

fig.tight_layout()
savefig(fig, "elbow_silhouette_train.png")
plt.close(fig)

# ---------- 4) Ajuste final con K óptimo y evaluación ----------
kmeans = safe_kmeans(n_clusters=best_k)
train_labels = kmeans.fit_predict(X_train_prep)
test_labels  = kmeans.predict(X_test_prep)

# Métricas internas
def internal_metrics(X, labels, subset_name=""):
    # muestreamos X si es enorme para silhouette (consistente con arriba)
    idx = sample_idx(X.shape[0], max_n=20000)
    sil = silhouette_score(X[idx], labels[idx]) if len(np.unique(labels)) > 1 else np.nan
    ch  = calinski_harabasz_score(X, labels)
    db  = davies_bouldin_score(X, labels)
    print(f"{subset_name} -> Silhouette: {sil:.3f} | Calinski-Harabasz: {ch:.1f} | Davies-Bouldin: {db:.3f}")
    return sil, ch, db

print("\nMétricas internas:")
sil_tr, ch_tr, db_tr = internal_metrics(X_train_prep, train_labels, "Train")
sil_te, ch_te, db_te = internal_metrics(X_test_prep,  test_labels,  "Test")

# Métricas externas (solo para evaluar relación con sexenio, NO para entrenar)
def external_metrics(y_true, labels, subset_name=""):
    ari = adjusted_rand_score(y_true, labels)
    ami = adjusted_mutual_info_score(y_true, labels, average_method='arithmetic')
    v   = v_measure_score(y_true, labels)
    print(f"{subset_name} (externas vs sexenio) -> ARI: {ari:.3f} | AMI: {ami:.3f} | V: {v:.3f}")
    return ari, ami, v

print("\nMétricas externas (con sexenio como referencia):")
ext_tr = external_metrics(y_train_ext, train_labels, "Train")
ext_te = external_metrics(y_test_ext,  test_labels,  "Test")

# ---------- 5) PCA para visualización ----------
def pca_scatter(X, labels, title, fname):
    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=RANDOM_SEED)
        XY = pca.fit_transform(X)
        exp = pca.explained_variance_ratio_.sum()
        subtitle = f"(var explicada 2D ≈ {exp*100:.1f}%)"
    else:
        XY = X
        subtitle = ""
    fig, ax = plt.subplots(figsize=(8,6))
    sc = ax.scatter(XY[:,0], XY[:,1], c=labels, s=12, alpha=0.6, cmap="tab10")
    ax.set_title(f"{title}\n{subtitle}")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Cluster")
    fig.tight_layout()
    savefig(fig, fname)
    plt.close(fig)

pca_scatter(X_train_prep, train_labels, f"PCA 2D – Train (K={best_k})", "pca_scatter_train.png")
pca_scatter(X_test_prep,  test_labels,  f"PCA 2D – Test  (K={best_k})", "pca_scatter_test.png")

# ---------- 6) Perfiles de clúster y matrices cluster×sexenio ----------
# Asignaciones con metadata original
train_assign = X_train.copy()
train_assign['cluster'] = train_labels
train_assign['sexenio'] = y_train_ext.values
train_assign['fecha']   = model_df.loc[train_mask, 'fecha'].values

test_assign = X_test.copy()
test_assign['cluster'] = test_labels
test_assign['sexenio'] = y_test_ext.values
test_assign['fecha']   = model_df.loc[test_mask,  'fecha'].values

# Perfiles en espacio original (medias por clúster)
def cluster_profiles(assign_df, name_prefix):
    # Solo numéricas para perfil cuantitativo
    prof = (assign_df
            .groupby('cluster')[['monto_asinh','lag1','roll3_mean','roll3_std','sin12','cos12']]
            .mean())
    prof.to_csv(os.path.join(OUT_DIR, f"{name_prefix}_cluster_profiles.csv"))
    # Heatmap
    fig, ax = plt.subplots(figsize=(8, max(3, 0.6*len(prof))))
    sns.heatmap(prof, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                cbar_kws={'shrink':0.7}, linewidths=0.4, linecolor='white')
    ax.set_title(f"Perfiles de Clúster – {name_prefix}")
    fig.tight_layout()
    savefig(fig, f"{name_prefix}_cluster_profiles_heatmap.png")
    plt.close(fig)

cluster_profiles(train_assign, "train")
cluster_profiles(test_assign,  "test")

# Matriz cluster × sexenio (conteos)
def cluster_vs_label_heatmap(assign_df, name_prefix):
    ctab = pd.crosstab(assign_df['cluster'], assign_df['sexenio'])
    fig, ax = plt.subplots(figsize=(6, max(3, 0.6*ctab.shape[0])))
    sns.heatmap(ctab, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Cluster × Sexenio – {name_prefix}")
    ax.set_xlabel("Sexenio"); ax.set_ylabel("Cluster")
    fig.tight_layout()
    savefig(fig, f"{name_prefix}_cluster_vs_sexenio.png")
    plt.close(fig)
    ctab.to_csv(os.path.join(OUT_DIR, f"{name_prefix}_cluster_vs_sexenio.csv"))

cluster_vs_label_heatmap(train_assign, "train")
cluster_vs_label_heatmap(test_assign,  "test")

# (Opcional) Serie temporal de promedio por clúster (test)
if not test_assign.empty:
    ts = (test_assign.groupby(['fecha','cluster'])['monto_asinh']
                    .mean()
                    .reset_index())
    fig, ax = plt.subplots(figsize=(12,5))
    for cl, g in ts.groupby('cluster'):
        ax.plot(g['fecha'], g['monto_asinh'], marker='.', linewidth=1.5, label=f"Cluster {cl}")
    ax.set_title("asinh(monto) promedio por clúster (Test)")
    ax.set_xlabel("Fecha"); ax.set_ylabel("asinh(monto)")
    ax.legend(ncol=2)
    fig.tight_layout()
    savefig(fig, "test_cluster_timeseries.png")
    plt.close(fig)

# ---------- 7) Exportar asignaciones detalladas ----------
train_assign.to_csv(os.path.join(OUT_DIR, "train_cluster_assignments.csv"), index=False)
test_assign.to_csv(os.path.join(OUT_DIR,  "test_cluster_assignments.csv"),  index=False)

# ---------- 8) Resumen en consola ----------
print("\n=== Resumen Práctica 7 (K-Means) ===")
print(f"K óptimo (silhouette en train): {best_k}")
print(f"Train  -> silhouette={sil_tr:.3f}, CH={ch_tr:.1f}, DB={db_tr:.3f} | ARI={ext_tr[0]:.3f}, AMI={ext_tr[1]:.3f}, V={ext_tr[2]:.3f}")
print(f"Test   -> silhouette={sil_te:.3f}, CH={ch_te:.1f}, DB={db_te:.3f} | ARI={ext_te[0]:.3f}, AMI={ext_te[1]:.3f}, V={ext_te[2]:.3f}")
print("\nArchivos en:", os.path.abspath(OUT_DIR))
print(" - elbow_silhouette_train.png")
print(" - pca_scatter_train.png, pca_scatter_test.png")
print(" - train_cluster_profiles.csv/.png, test_cluster_profiles.csv/.png")
print(" - train_cluster_vs_sexenio.csv/.png, test_cluster_vs_sexenio.csv/.png")
print(" - test_cluster_timeseries.png (si aplica)")
print(" - train_cluster_assignments.csv, test_cluster_assignments.csv")
