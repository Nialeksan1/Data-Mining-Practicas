# Ingreso, Gasto y Financiamiento Público (2012–2024) — Proyecto de Minería de Datos

**Curso:** Minería de Datos
**Autor:** _Nineck-Aleksander Hournou_
**Dataset limpio (base de trabajo):** `../Ingreso_Gasto_Financiamiento_Pena_AMLO_2012-2024.csv`
**Origen (bruto):** `../Ingreso, Gasto y Financiamiento Público ( 2011 - Actual ) de SHCP.csv`

> Este repositorio documenta **9 prácticas** completas: limpieza, estadística descriptiva, visualización, pruebas estadísticas, modelos lineales, clasificación, clustering, forecasting y análisis de texto.
> El README explica **qué hace** cada script, **cómo ejecutarlo** y **qué archivos** genera.

---

## Requisitos

### Librerías principales

```bash
pip install -U pandas numpy matplotlib seaborn scikit-learn statsmodels scipy wordcloud nltk
```

> Para NLTK (stopwords en español) la primera vez:
>
> ```python
> import nltk
> nltk.download("stopwords")
> ```

---

## Estructura de archivos

```
.
├── Practica1/
├── Practica2/
├── Practica3/
├── Practica4/
├── Practica5/
├── Practica6/
├── Practica7/
├── Practica8/
├── Practica9/
└── README.md
```

---

## Cómo ejecutar cada práctica

> **Nota de rutas:**
>
> - La **Práctica 1** lee el **archivo bruto** y guarda el limpio como `./Ingreso_Gasto_Financiamiento_Pena_AMLO_2012-2024.csv`.
> - **Prácticas 2–9** consumen el **archivo limpio** anterior con esa misma ruta.

---

### 1) Data Cleaning — semana 4

**Script:** (dentro de _Practica1_)

- Lee: `../Ingreso, Gasto y Financiamiento Público ( 2011 - Actual ) de SHCP.csv`
- Crea `mes_num`, construye `fecha`, **filtra** el rango temporal (dic-2012 a sep-2024) y guarda:

  - **Salida:** `../Ingreso_Gasto_Financiamiento_Pena_AMLO_2012-2024.csv`

- Imprime **resumen básico** (`describe`) de columnas clave.

**Ejecutar:**

```bash
py Practica1/limpiarDataset.py
```

---

### 2) Descriptive Statistics — semana 5

**Script:** (dentro de _Practica2_)

- Carga el **dataset limpio**.
- Reporta forma, memoria aproximada, tipos de datos agrupados, **nulos por columna**, `describe()` de numéricas y conteos de categóricas.
- **Agrupa** por entidades: `subtema`, `sector`, `ciclo` y calcula estadísticos.
- Genera un **Diagrama ER** en Mermaid y lo guarda en `er_diagram.md`.
- Exporta CSVs de **valores únicos** (con conteos) para varias columnas.

**Entradas/Salidas destacadas:**

- Entrada: `../Ingreso_Gasto_Financiamiento_Pena_AMLO_2012-2024.csv`
- Salidas: `er_diagram.md`, `*_unicos.csv`

**Ejecutar:**

```bash
py Practica2/datosDescriptivosIniciales.py
```

---

### 3) Data Visualization — semana 6

**Script:** (dentro de _Practica3_)

- Enfocado en **primer año** de cada sexenio:

  - Peña: dic-2012 a nov-2013
  - AMLO: dic-2018 a nov-2019

- Construye `mes_relativo` (1=Dic, …, 12=Nov).
- Genera al menos **5 tipos** de visualizaciones:

  - **Líneas**: suma mensual por subtema en el primer año.
  - **Barras**: suma total por sexenio (primer año).
  - **Boxplots**: distribución de `monto` por sexenio.
  - **Pie**: proporción de registros por subtema.
  - **Scatter**: `monto` vs `mes_relativo`.
  - **Barras horizontales**: Top 5 sectores por suma (por sexenio).

- Emplea **escala `symlog`** cuando hay negativos y _outliers_.
- **Guarda** las figuras (`.png`) en el directorio de trabajo.

**Entradas/Salidas destacadas:**

- Entrada: `../Ingreso_Gasto_Financiamiento_Pena_AMLO_2012-2024.csv`
- Salidas: `line_12meses_subtema.png`, `bar_compar_12meses.png`, `boxplots_12meses.png`, `pies_subtema_12meses.png`, `scatter_12meses.png`, `barras_top_12meses.png`

**Ejecutar:**

```bash
py Practica3/dataVisualization.py
```

---

### 4) Statistic Test — semana 7

**Script:** (dentro de _Practica4_)

- Construye panel mensual por `fecha×sexenio×subtema` y por `fecha×sexenio` (totales).
- Aplica transformaciones (`asinh`) para estabilizar escala.
- **ANOVA de dos vías**: `monto_asinh ~ C(sexenio) * C(subtema)` (Type II SS).
- **Pruebas globales** Peña vs AMLO:

  - Welch t-test (paramétrico) en `asinh`.
  - Mann–Whitney U (no paramétrico) en `monto` original.

- **Kruskal–Wallis** dentro de cada sexenio (por subtema).
- **Welch t-tests** por subtema con **FDR (Benjamini–Hochberg)**.

**Entradas/Salidas destacadas:**

- Entrada: `../Ingreso_Gasto_Financiamiento_Pena_AMLO_2012-2024.csv`
- Salidas: `salidas_tests/anova_two_way.csv`, `test_global_sexenio.csv`, `kruskal_por_sexenio.csv`, `ttests_subtema_fdr.csv`

**Ejecutar:**

```bash
py Practica4/ANOVA2.py
```

---

### 5) Linear Models + correlation — semana 9

**Script:** (dentro de _Practica5_)

- Panel mensual por `fecha×sexenio×subtema` con variables:

  - `asinh(monto)`, `t` (tendencia), estacionalidad `sin12/cos12`, `lag1`, y **dummies** de `sexenio`/`subtema` + **interacción**.

- **Heatmaps** de correlación (Pearson y Spearman).
- Ajuste con **OLS** y **errores robustos HC3**.
- _Split_ temporal (por defecto, inicio **2022**; con **fallback 80/20** si hace falta).
- **Gráficas**:

  - Predicho vs Real (test), Residuales vs Ajustados (test),
  - Líneas Real vs Predicho por subtema (test),
  - Barras de coeficientes con IC95%.

- **Exporta** tabla de coeficientes y predicciones.

**Entradas/Salidas destacadas:**

- Entrada: `../Ingreso_Gasto_Financiamiento_Pena_AMLO_2012-2024.csv`
- Salidas: `salidas_practica5/corr_heatmap_*.png`, `pred_vs_real_test.png`, `resid_vs_fitted_test.png`, `line_real_vs_pred_<subtema>_test.png`, `coeficientes_bar_ic.png`, `coeficientes_ols_hc3.csv`, `predicciones_test.csv`

**Ejecutar:**

```bash
py Practica5/linearModel.py
```

---

### 6) Data classification (KNN) — semana 10

**Script:** (dentro de _Practica6_)

- Construye panel por `fecha×subtema×sector` y features:

  - `asinh(monto)`, `lag1`, `roll3_mean`, `roll3_std`, `sin12`, `cos12`, `subtema`, `sector`.

- **Objetivo**: `sexenio`.
- Pipeline con **preprocesamiento** (imputación, escalado, One-Hot) y **KNN**.
- Validación cruzada **TimeSeriesSplit**; búsqueda de hiperparámetros (`n_neighbors`, `weights`, `p`).
- _Hold-out_ temporal (por defecto, **2022**).
- **Métricas y gráficas**:

  - Matriz de confusión (`.png`),
  - Curva ROC (`.png`),
  - Predicciones detalladas (`.csv`).

**Entradas/Salidas destacadas:**

- Entrada: `../Ingreso_Gasto_Financiamiento_Pena_AMLO_2012-2024.csv`
- Salidas: `salidas_practica6/confusion_matrix_test.png`, `roc_curve_test.png`, `predicciones_knn_test.csv`

**Ejecutar:**

```bash
py Practica6/kNeighbors.py
```

> **Nota metodológica:** el _hold-out_ 2022–2024 puede contener mayormente una sola clase temporal; el script ya maneja tipos de etiquetas y curvas ROC de manera robusta.

---

### 7) Data clustering (K-means) — semana 12

**Script:** (dentro de _Practica7_)

- Mismo panel por `fecha×subtema×sector` y mismas features numéricas + categóricas (One-Hot).
- Selección de **K** con **elbow** e **índice silhouette** (muestrado si N es grande).
- Ajuste final con **KMeans** y evaluación:

  - **Interna:** silhouette, Calinski–Harabasz, Davies–Bouldin.
  - **Externa (no supervisada):** relación con `sexenio` (ARI, AMI, V-measure).

- **PCA 2D** para visualización y **perfiles de clúster** (medias).
- **Exporta** asignaciones y perfiles.

**Entradas/Salidas destacadas:**

- Entrada: `../Ingreso_Gasto_Financiamiento_Pena_AMLO_2012-2024.csv`
- Salidas: `salidas_practica7/elbow_silhouette_train.png`, `pca_scatter_*.png`, `*_cluster_profiles.csv/.png`, `*_cluster_vs_sexenio.csv/.png`, `test_cluster_timeseries.png`, `*_cluster_assignments.csv`

**Ejecutar:**

```bash
py Practica7/kMeans.py
```

---

### 8) Forecasting (Regresión lineal) — semana 13

**Script:** (dentro de _Practica8_)

- Serie **agregada mensual** (total país).
- Features: `t`, `sin12`, `cos12`, `lag1`, `lag12` sobre objetivo transformado con **asinh** (y su inversa para volver a la escala original).
- _Hold-out_ temporal desde **2022-01-01**.
- **Predicción dinámica** (evita fugas usando solo información pasada).
- **Forecast** de **H** meses hacia delante.
- **Exporta** CSV combinado con real, predicho y forecast; y **gráficas** de serie completa, residuales y _scatter_ real vs predicho (en asinh).

**Entradas/Salidas destacadas:**

- Entrada: `../Ingreso_Gasto_Financiamiento_Pena_AMLO_2012-2024.csv`
- Salidas: `salidas_practica8/forecast_linear_regression.csv`, `forecast_lineal_serie.png`, `forecast_residuales_test.png`, `forecast_scatter_test_asinh.png`

**Ejecutar:**

```bash
py Practica8/forecasting.py
```

---

### 9) Text analysis (Word Cloud) — semana 14

**Script:** (dentro de _Practica9_)

- Construye **corpus** concatenando columnas textuales (`nombre`, `subtema`, `sector`, `tema`, `clave_de_concepto`).
- Normaliza: minúsculas, sin acentos, sin puntuación ni números, limpieza por **stopwords** (ES + dominio fiscal).
- Usa `CountVectorizer` (con _fallback_ manual) para **frecuencias** de **unigramas** y **bigramas**.
- Genera **nubes** globales, por **sexenio** y por **subtema**; además **top terms** en CSV.
- Maneja `WordCloud` (con _fallback_ manual si no está disponible).

**Entradas/Salidas destacadas:**

- Entrada: `../Ingreso_Gasto_Financiamiento_Pena_AMLO_2012-2024.csv`
- Salidas:

  - Global: `wc_global_unigramas.png`, `wc_global_bigramas.png`, `top100_global_unigramas.csv`, `top100_global_bigramas.csv`
  - Por sexenio: `wc_peña_nieto_*.png`, `wc_amlo_*.png`, `top100_peña_nieto_*.csv`, `top100_amlo_*.csv`
  - Por subtema: `wc_subtema_*.png`, `top50_subtema_*.csv`

**Ejecutar:**

```bash
py Practica9/textAnalysis.py
```
