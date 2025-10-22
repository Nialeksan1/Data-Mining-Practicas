import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests


# Mostrar floats “normales” en pandas (ej. 1,234,567.89)
pd.options.display.float_format = '{:,.2f}'.format

# Evitar notación científica en NumPy (prints, arrays)
np.set_printoptions(suppress=True)


# ==============
# PARÁMETROS
# ==============
FILE_PATH = "../Ingreso_Gasto_Financiamiento_Pena_AMLO_2012-2024.csv"
OUT_DIR   = "./salidas_tests"
os.makedirs(OUT_DIR, exist_ok=True)

CUTOFF_DATE = pd.to_datetime("2018-12-01")  # Frontera Peña vs AMLO
RNG = np.random.default_rng(42)


# ==============
# UTILIDADES
# ==============
def ensure_mes_num(df: pd.DataFrame) -> pd.DataFrame:
    mes_map = {'Enero':1,'Febrero':2,'Marzo':3,'Abril':4,'Mayo':5,'Junio':6,
               'Julio':7,'Agosto':8,'Septiembre':9,'Octubre':10,'Noviembre':11,'Diciembre':12}
    if 'mes_num' not in df.columns:
        df['mes_num'] = df['mes'].map(mes_map)
    return df

def ensure_fecha(df: pd.DataFrame) -> pd.DataFrame:
    # Construir YYYY-MM-01 y normalizar a inicio de mes (sin 'MS')
    df['fecha'] = pd.to_datetime(
        df['ciclo'].astype(str) + '-' + df['mes_num'].astype(str) + '-01',
        errors='coerce'
    )
    df['fecha'] = df['fecha'].dt.to_period('M').dt.to_timestamp()  # inicio de mes
    return df.dropna(subset=['fecha'])

def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df['monto'] = pd.to_numeric(df['monto'], errors='coerce')
    return df.dropna(subset=['monto'])

def asinh_series(s: pd.Series) -> pd.Series:
    return np.arcsinh(s.astype(float))

def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    """Tamaño de efecto Hedges' g (corrección de pequeño tamaño)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    if vx == 0 and vy == 0:
        return 0.0
    sp = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2)) if (nx+ny-2)>0 else np.nan
    if sp == 0 or np.isnan(sp):
        return np.nan
    g = (np.mean(x) - np.mean(y)) / sp
    J = 1 - (3 / (4*(nx+ny) - 9)) if (4*(nx+ny)-9) != 0 else 1.0
    return g * J

def partial_eta_sq(anova_tbl: pd.DataFrame, effect: str) -> float:
    """η² parcial para un efecto en ANOVA (SS_effect / (SS_effect + SS_residual))."""
    if effect not in anova_tbl.index or 'Residual' not in anova_tbl.index:
        return np.nan
    ss_e = anova_tbl.loc[effect, 'sum_sq']
    ss_r = anova_tbl.loc['Residual', 'sum_sq']
    return float(ss_e / (ss_e + ss_r)) if (ss_e + ss_r) > 0 else np.nan

def epsilon_sq_kruskal(H, k, N) -> float:
    """Epsilon² para Kruskal–Wallis (tamaño de efecto)."""
    if N <= 1:
        return np.nan
    return (H - (k - 1)) / (N - 1)

def print_header(msg: str):
    print("\n" + "="*len(msg))
    print(msg)
    print("="*len(msg))


# ==============
# 1) CARGA Y PREPARACIÓN
# ==============
print(f"Cargando dataset: {FILE_PATH}")
df = pd.read_csv(FILE_PATH, low_memory=False)
print(f"✓ Datos: {df.shape[0]:,} filas × {df.shape[1]} columnas")

# Validaciones básicas
for col in ['ciclo','mes','monto','subtema']:
    if col not in df.columns:
        raise ValueError(f"Falta columna requerida: {col}")

df = ensure_mes_num(df)
df = ensure_fecha(df)
df = ensure_numeric(df)

# Etiqueta de sexenio
df['sexenio'] = np.where(df['fecha'] < CUTOFF_DATE, 'Peña Nieto', 'AMLO')

# Panel mensual por subtema
monthly_subtema = (
    df.groupby(['fecha','sexenio','subtema'], observed=True, as_index=False)['monto']
      .sum()
      .sort_values('fecha')
)
monthly_subtema['monto_asinh'] = asinh_series(monthly_subtema['monto'])

# Panel mensual total por sexenio
monthly_total = (
    df.groupby(['fecha','sexenio'], observed=True, as_index=False)['monto']
      .sum()
      .sort_values('fecha')
)
monthly_total['monto_asinh'] = asinh_series(monthly_total['monto'])

print("Ejemplo monthly_subtema:")
print(monthly_subtema.head(6))


# ==============
# 2) TWO-WAY ANOVA: monto_asinh ~ sexenio * subtema
# ==============
print_header("ANOVA de dos vías (monto_asinh ~ C(sexenio) * C(subtema))")
model = smf.ols('monto_asinh ~ C(sexenio) * C(subtema)', data=monthly_subtema).fit()
anova_tbl = anova_lm(model, typ=2)  # Type II SS (a prueba de desbalance)
print(anova_tbl)

# Tamaños de efecto (η² parcial)
eta_sex   = partial_eta_sq(anova_tbl, 'C(sexenio)')
eta_sub   = partial_eta_sq(anova_tbl, 'C(subtema)')
eta_inter = partial_eta_sq(anova_tbl, 'C(sexenio):C(subtema)')
print(f"\nη² parcial – sexenio: {eta_sex:.3f} | subtema: {eta_sub:.3f} | interacción: {eta_inter:.3f}")

# Guardar
anova_tbl.to_csv(os.path.join(OUT_DIR, "anova_two_way.csv"))


# ==============
# 3) TEST GLOBAL: Peña vs AMLO (Welch t-test y Mann–Whitney)
# ==============
print_header("Test global Peña vs AMLO (sobre series mensuales totales)")
x = monthly_total.loc[monthly_total['sexenio']=='Peña Nieto', 'monto_asinh'].values
y = monthly_total.loc[monthly_total['sexenio']=='AMLO',       'monto_asinh'].values

# Welch t-test (paramétrico)
t_stat, t_p = stats.ttest_ind(x, y, equal_var=False)
g_eff = hedges_g(x, y)
print(f"Welch t-test (asinh): t = {t_stat:.2f}, p = {t_p:.3e} | Hedges' g = {g_eff:.3f}")

# Mann–Whitney U (no paramétrico) sobre monto bruto agregado
x_raw = monthly_total.loc[monthly_total['sexenio']=='Peña Nieto', 'monto'].values
y_raw = monthly_total.loc[monthly_total['sexenio']=='AMLO',       'monto'].values
u_stat, u_p = stats.mannwhitneyu(x_raw, y_raw, alternative='two-sided')
# Efecto r aproximado: z / sqrt(N)
N_tot = len(x_raw) + len(y_raw)
# Aproximación z desde U
mu_U = (len(x_raw)*len(y_raw))/2
sigma_U = np.sqrt(len(x_raw)*len(y_raw)*(len(x_raw)+len(y_raw)+1)/12)
z = (u_stat - mu_U)/sigma_U if sigma_U > 0 else np.nan
r_eff = z / np.sqrt(N_tot) if N_tot > 0 else np.nan
print(f"Mann–Whitney U (monto): U = {u_stat:.0f}, p = {u_p:.3e} | r ≈ {r_eff:.3f}")

# Guardar resumen global
pd.DataFrame({
    'test': ['Welch_t_asinh','MannWhitney_monto'],
    'statistic': [t_stat, u_stat],
    'p_value': [t_p, u_p],
    'effect_size': [g_eff, r_eff]
}).to_csv(os.path.join(OUT_DIR, "test_global_sexenio.csv"), index=False)


# ==============
# 4) KRUSKAL–WALLIS por subtema (dentro de cada sexenio)
# ==============
print_header("Kruskal–Wallis por subtema (dentro de cada sexenio)")
kw_rows = []
for sex in ['Peña Nieto', 'AMLO']:
    d = monthly_subtema[monthly_subtema['sexenio']==sex]
    groups = [grp['monto'].values for _, grp in d.groupby('subtema')]
    labels = d['subtema'].unique()
    if len(groups) >= 2:
        H, p = stats.kruskal(*groups)
        # Epsilon² como tamaño de efecto
        N = sum(len(g) for g in groups)
        eps2 = epsilon_sq_kruskal(H, k=len(groups), N=N)
        print(f"{sex}: H = {H:.2f}, p = {p:.3e} | ε² = {eps2:.3f}")
        kw_rows.append({'sexenio':sex, 'H':H, 'p_value':p, 'epsilon_sq':eps2, 'k_grupos':len(groups), 'N':N})

pd.DataFrame(kw_rows).to_csv(os.path.join(OUT_DIR, "kruskal_por_sexenio.csv"), index=False)


# ==============
# 5) WELCH t-tests por SUBTEMA (Peña vs AMLO) + FDR
# ==============
print_header("Welch t-tests por subtema (Peña vs AMLO) con FDR")
tt_rows = []
for subt, g in monthly_subtema.groupby('subtema'):
    x = g.loc[g['sexenio']=='Peña Nieto','monto_asinh'].values
    y = g.loc[g['sexenio']=='AMLO','monto_asinh'].values
    if len(x) > 1 and len(y) > 1:
        t, p = stats.ttest_ind(x, y, equal_var=False)
        gsize = hedges_g(x, y)
        tt_rows.append({'subtema':subt, 't_stat': t, 'p_value': p, "hedges_g": gsize, "n_pena": len(x), "n_amlo": len(y)})

tt_df = pd.DataFrame(tt_rows).sort_values('p_value')
if not tt_df.empty:
    # FDR (Benjamini–Hochberg)
    rej, p_adj, *_ = multipletests(tt_df['p_value'].values, alpha=0.05, method='fdr_bh')
    tt_df['p_adj_fdr'] = p_adj
    tt_df['rej_fdr_5%'] = rej
    print(tt_df[['subtema','t_stat','p_value','p_adj_fdr','hedges_g','n_pena','n_amlo']])
    tt_df.to_csv(os.path.join(OUT_DIR, "ttests_subtema_fdr.csv"), index=False)
else:
    print("No hubo subtemas con datos suficientes para el t-test.")


print("\n✓ Resultados guardados en:", os.path.abspath(OUT_DIR))
print("   - anova_two_way.csv")
print("   - test_global_sexenio.csv")
print("   - kruskal_por_sexenio.csv")
print("   - ttests_subtema_fdr.csv")
print("\nInterpretación:")
print("  • Si el efecto 'C(sexenio)' en ANOVA es significativo (p<0.05) y/o los tests globales")
print("    (Welch/Mann–Whitney) dan p<0.05, se puede concluir que el etiquetado (Peña vs AMLO)")
print("    está asociado a diferencias en el nivel central de los montos (con η²/g/r como magnitud).")

