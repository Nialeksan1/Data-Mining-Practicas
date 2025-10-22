import os, re, unicodedata, random, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIGURACIÓN
# =========================
FILE_PATH = "../Ingreso_Gasto_Financiamiento_Pena_AMLO_2012-2024.csv"  # <- path único solicitado
OUT_DIR   = "./salidas_practica9"
os.makedirs(OUT_DIR, exist_ok=True)

CUTOFF_DATE  = pd.to_datetime("2018-12-01")  # frontera Peña vs AMLO
RANDOM_SEED  = 42
TOP_GLOBAL   = 100
TOP_SUBTEMA  = 50
MAX_SUBTEMAS = None    # None = todos o un entero para limitar
MIN_DF_UNI   = 5       # frecuencia mínima (documentos) para unigramas
MIN_DF_BI    = 5       # para bigramas
MAX_FEATURES = 5000    # tope de vocabulario (None = sin tope)

np.random.seed(RANDOM_SEED)
plt.style.use('seaborn-v0_8-darkgrid')

# =========================
# STOPWORDS (ES + dominio)
# =========================
SPANISH_SW = {
    'de','la','que','el','en','y','a','los','del','se','las','por','un','para','con','no','una','su','al','lo','como',
    'mas','pero','sus','le','ya','o','este','si','porque','esta','entre','cuando','muy','sin','sobre','tambien','me',
    'hasta','hay','donde','quien','desde','todo','nos','durante','todos','uno','les','ni','contra','otros','ese','eso',
    'ante','ellos','e','esto','antes','algunos','unos','yo','otro','otras','otra','tanto','esa','estos',
    'mucho','quienes','nada','muchos','cual','poco','ella','estar','estas','algunas','algo','nosotros','mi','mis','tu',
    'te','ti','tus','ellas','nosotras','vosotros','vosotras','os','mio','mia','mios','mias','tuyo','tuya','tuyos',
    'tuyas','suyo','suya','suyos','suyas','nuestro','nuestra','nuestros','nuestras','vuestro','vuestra','vuestros',
    'vuestras','esos','esas','estoy','estas','esta','estamos','estan','ser','es','son','fue','era','han',
    'he','has','ha','haya','haber','va','van','ir',
    # meses
    'enero','febrero','marzo','abril','mayo','junio','julio','agosto','septiembre','octubre','noviembre','diciembre',
}
DOMAIN_SW = {
    'estadisticas','finanzas','publicas','publica','miles','pesos','mensual','pagado','devengado','flujo',
    'ingreso','ingresos','gasto','gastos','financiamiento','resultado','operacion','variacion','disponibilidades',
    'sector','ambito','federal','gobierno','nacional','servicio','servicios','instituto','comision','empresa',
    'otros','otras','concepto','nombre','tema','subtema','periodo','frecuencia','difusion','base','registro',
    'unidad','unidades','medida','mexico','mexicana','mexicanas'
}
STOPWORDS = SPANISH_SW | DOMAIN_SW

# =========================
# UTILIDADES
# =========================
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = strip_accents(s)
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)  # quita puntuación
    s = re.sub(r"\d+", " ", s)                        # quita números
    s = re.sub(r"\s+", " ", s).strip()
    return s

def ensure_mes_num(df: pd.DataFrame) -> pd.DataFrame:
    if 'mes_num' not in df.columns and 'mes' in df.columns:
        mes_map = {'Enero':1,'Febrero':2,'Marzo':3,'Abril':4,'Mayo':5,'Junio':6,
                   'Julio':7,'Agosto':8,'Septiembre':9,'Octubre':10,'Noviembre':11,'Diciembre':12}
        df['mes_num'] = df['mes'].map(mes_map)
    return df

def ensure_fecha_and_sexenio(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Crear/convertir 'fecha'
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce', dayfirst=False, infer_datetime_format=True)
        if df['fecha'].isna().mean() > 0.2:
            df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce', dayfirst=True, infer_datetime_format=True)
    else:
        df = ensure_mes_num(df)
        if {'ciclo','mes_num'} <= set(df.columns):
            df['fecha'] = pd.to_datetime(df['ciclo'].astype(str) + '-' + df['mes_num'].astype(str) + '-01', errors='coerce')
        elif 'periodo_inicio' in df.columns:
            df['fecha'] = pd.to_datetime(df['periodo_inicio'].astype(str) + "-01", errors='coerce')
        else:
            raise ValueError("No se puede construir 'fecha': faltan ('fecha') o ('ciclo'+'mes/mes_num') o ('periodo_inicio').")
    # 2) Normalizar a inicio de mes
    df['fecha'] = df['fecha'].dt.to_period('M').dt.to_timestamp()
    df = df.dropna(subset=['fecha'])
    # 3) Etiquetar sexenio
    df['sexenio'] = np.where(df['fecha'] < CUTOFF_DATE, 'Peña Nieto', 'AMLO')
    return df

def join_text_cols(df, cols=('nombre','subtema','sector','tema','clave_de_concepto')):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        cols = [c for c in df.columns if df[c].dtype == 'object'][:4]  # fallback
    return df[cols].astype(str).agg(" ".join, axis=1)

# Frecuencias con CountVectorizer (o fallback manual)
try:
    from sklearn.feature_extraction.text import CountVectorizer
    SK_HAS_CV = True
except Exception:
    SK_HAS_CV = False

def frequencies_from_series(text_series: pd.Series, stopwords, ngram_range=(1,1), min_df=5, max_features=None):
    if SK_HAS_CV:
        cv = CountVectorizer(
            stop_words=list(stopwords),
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features
        )
        X = cv.fit_transform(text_series.values)
        vocab = cv.get_feature_names_out()
        freqs = np.asarray(X.sum(axis=0)).ravel()
        freq_dict = dict(zip(vocab, freqs))
    else:
        # Fallback sin sklearn (min_df ≈ conteo mínimo)
        from collections import Counter
        tokenized = []
        for s in text_series.values:
            toks = [t for t in s.split() if t not in stopwords and len(t) >= 3]
            tokenized.append(toks)
        cnt = Counter()
        if ngram_range == (1,1):
            for toks in tokenized: cnt.update(toks)
        elif ngram_range == (2,2):
            for toks in tokenized: cnt.update([" ".join(toks[i:i+2]) for i in range(len(toks)-1)])
        freq_dict = {k:v for k,v in cnt.items() if v >= min_df}
        if max_features:
            freq_dict = dict(sorted(freq_dict.items(), key=lambda kv: kv[1], reverse=True)[:max_features])
    return dict(sorted(freq_dict.items(), key=lambda kv: kv[1], reverse=True))

# WordCloud (o fallback manual)
try:
    from wordcloud import WordCloud
    HAS_WC = True
except Exception:
    HAS_WC = False

from matplotlib import font_manager
FONT_PATH = font_manager.findfont('DejaVu Sans')

def draw_text_cloud_from_freq(freq_dict, title, fname, width=1600, height=900, max_words=250, seed=RANDOM_SEED):
    items = sorted(freq_dict.items(), key=lambda kv: kv[1], reverse=True)[:max_words]
    if not items:
        return None
    freqs = np.array([v for _, v in items], dtype=float)
    sizes = 10 + 50 * (freqs - freqs.min()) / (freqs.max() - freqs.min() + 1e-9)
    random.seed(seed)
    cols = int(math.sqrt(len(items))) + 5
    rows = int(math.ceil(len(items) / cols))
    xs, ys = np.linspace(0.1, 0.9, cols), np.linspace(0.9, 0.1, rows)
    positions = [(x, y) for y in ys for x in xs][:len(items)]
    random.shuffle(positions)
    fig = plt.figure(figsize=(width/100, height/100))
    ax = plt.gca(); ax.axis('off')
    for (word, _), sz, (x, y) in zip(items, sizes, positions):
        ax.text(x, y, word, fontsize=float(sz), ha='center', va='center', transform=ax.transAxes)
    ax.set_title(title)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, fname)
    fig.savefig(out_path, dpi=220, bbox_inches="tight"); plt.close(fig)
    return out_path

def wordcloud_from_freq(freq_dict, title, fname, width=1600, height=900, max_words=300):
    if not freq_dict:
        print(f"[AVISO] Sin términos para '{title}'.")
        return None
    if HAS_WC:
        try:
            wc = WordCloud(
                width=width, height=height, background_color='white',
                max_words=max_words, prefer_horizontal=0.9, collocations=False,
                random_state=RANDOM_SEED, font_path=FONT_PATH
            ).generate_from_frequencies(freq_dict)
            fig = plt.figure(figsize=(width/100, height/100))
            ax = plt.gca()
            ax.imshow(wc, interpolation='bilinear'); ax.axis('off'); ax.set_title(title)
            fig.tight_layout(pad=0)
            out_path = os.path.join(OUT_DIR, fname)
            fig.savefig(out_path, dpi=220, bbox_inches='tight'); plt.close(fig)
            print(f"✓ Nube: {out_path}")
            return out_path
        except Exception as e:
            print("[Aviso] WordCloud falló, uso fallback manual:", e)
    out_path = draw_text_cloud_from_freq(freq_dict, title, fname, width, height, max_words)
    if out_path:
        print(f"✓ Nube (fallback): {out_path}")
    return out_path

def export_top_terms(freq_dict, top_k, fname):
    items = list(freq_dict.items())[:top_k]
    df = pd.DataFrame(items, columns=['term','freq'])
    out_path = os.path.join(OUT_DIR, fname)
    df.to_csv(out_path, index=False)
    print(f"✓ CSV: {out_path}")
    return out_path

# =========================
# 1) CARGA
# =========================
print(f"Cargando: {FILE_PATH}")
try:
    df = pd.read_csv(FILE_PATH, low_memory=False)
except UnicodeDecodeError:
    df = pd.read_csv(FILE_PATH, low_memory=False, encoding='latin-1')

print(f"✓ Datos: {df.shape[0]:,} filas × {df.shape[1]} columnas")

# =========================
# 2) FECHA & SEXENIO
# =========================
df = ensure_fecha_and_sexenio(df)

# =========================
# 3) CORPUS
# =========================
text_cols_pref = ['nombre','subtema','sector','tema','clave_de_concepto']
df['texto'] = join_text_cols(df, text_cols_pref).apply(normalize_text)

# Limpieza adicional con stopwords por regex
if STOPWORDS:
    pattern = r'\b(' + '|'.join(re.escape(sw) for sw in sorted(STOPWORDS)) + r')\b'
    df['texto'] = (df['texto']
                   .str.replace(pattern, ' ', regex=True)
                   .str.replace(r'\s+', ' ', regex=True)
                   .str.strip())

# =========================
# 4) GLOBAL
# =========================
freq_uni_global = frequencies_from_series(
    df['texto'], STOPWORDS, ngram_range=(1,1),
    min_df=MIN_DF_UNI, max_features=MAX_FEATURES
)
freq_bi_global  = frequencies_from_series(
    df['texto'], STOPWORDS, ngram_range=(2,2),
    min_df=MIN_DF_BI,  max_features=MAX_FEATURES
)

wordcloud_from_freq(freq_uni_global, "Word Cloud – Global (Unigramas)", "wc_global_unigramas.png")
export_top_terms(freq_uni_global, TOP_GLOBAL, "top100_global_unigramas.csv")

wordcloud_from_freq(freq_bi_global,  "Word Cloud – Global (Bigramas)",  "wc_global_bigramas.png")
export_top_terms(freq_bi_global,  TOP_GLOBAL, "top100_global_bigramas.csv")

# =========================
# 5) POR SEXENIO
# =========================
for sex in sorted(df['sexenio'].dropna().unique()):
    dsex = df[df['sexenio'] == sex]
    if dsex.empty:
        continue
    freq_uni = frequencies_from_series(dsex['texto'], STOPWORDS, (1,1),
                                       min_df=max(3, MIN_DF_UNI//2), max_features=MAX_FEATURES)
    freq_bi  = frequencies_from_series(dsex['texto'], STOPWORDS, (2,2),
                                       min_df=max(3, MIN_DF_BI//2),  max_features=MAX_FEATURES)
    safe = strip_accents(sex).lower().replace(' ', '_')
    wordcloud_from_freq(freq_uni, f"Word Cloud – {sex} (Unigramas)", f"wc_{safe}_unigramas.png")
    export_top_terms(freq_uni, TOP_GLOBAL, f"top100_{safe}_unigramas.csv")
    wordcloud_from_freq(freq_bi,  f"Word Cloud – {sex} (Bigramas)",  f"wc_{safe}_bigramas.png")
    export_top_terms(freq_bi,  TOP_GLOBAL, f"top100_{safe}_bigramas.csv")

# =========================
# 6) POR SUBTEMA
# =========================
if 'subtema' in df.columns:
    subtemas = [s for s in df['subtema'].dropna().unique().tolist() if isinstance(s, str)]
    if isinstance(MAX_SUBTEMAS, int):
        subtemas = subtemas[:MAX_SUBTEMAS]
    for subt in subtemas:
        dsub = df[df['subtema'] == subt]
        if dsub.empty:
            continue
        freq_uni = frequencies_from_series(dsub['texto'], STOPWORDS, (1,1),
                                           min_df=max(3, MIN_DF_UNI//2),
                                           max_features=MAX_FEATURES//2 if MAX_FEATURES else None)
        if not freq_uni:
            continue
        safe = strip_accents(re.sub(r'\W+', '_', subt)).lower().strip('_')
        wordcloud_from_freq(freq_uni, f"Word Cloud – Subtema: {subt}", f"wc_subtema_{safe}.png")
        export_top_terms(freq_uni, TOP_SUBTEMA, f"top50_subtema_{safe}.csv")

print("\n✓ Listo. Archivos generados en:", os.path.abspath(OUT_DIR))
print("   - wc_global_unigramas.png / wc_global_bigramas.png")
print("   - wc_peña_nieto_*.png, wc_amlo_*.png (si aplica)")
print("   - wc_subtema_*.png")
print("   - top100_*_unigramas.csv / top100_*_bigramas.csv / top50_subtema_*.csv")
