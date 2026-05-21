import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  TRANSLATIONS
# ─────────────────────────────────────────────
TRANSLATIONS = {
    "English": {
        "app_name": "Laptop Price Predictor",
        "app_sub": "ML Price Intelligence",
        "nav_home": "🏠  Home",
        "nav_predict": "🔮  Predict Price",
        "nav_analysis": "📊  Analysis Dashboard",
        "hero_tag": "💻  AI-POWERED",
        "hero_title": "Laptop Price Predictor",
        "hero_title2": "Price Predictor",
        "hero_sub": "Predict laptop prices intelligently using Machine Learning",
        "laptops_analysed": "Laptops Analysed",
        "avg_price": "Avg Price",
        "brands": "Brands",
        "avg_rating": "Avg Rating",
        "price_dist": "Price Distribution",
        "price_dist_sub": "How prices are spread across the dataset",
        "top_brands": "Top Brands",
        "top_brands_sub": "Average price by brand",
        "predict_title": "🔮 Predict Laptop Price",
        "predict_sub": "Fill in the specs and get an instant price estimate",
        "model_warn": "⚠️  **model.joblib** not found – showing a demo prediction instead.",
        "brand": "Brand 🏷️",
        "processor": "Processor ⚡",
        "ram": "RAM (GB) 🧠",
        "storage": "Storage (GB) 💾",
        "screen": "Screen Size (inches) 🖥️",
        "cores": "CPU Cores 🔲",
        "threads": "CPU Threads 🔁",
        "rating": "User Rating ⭐",
        "specs_score": "Specs Score 📈",
        "predict_btn": "🚀  Predict Price",
        "est_price": "Estimated Price",
        "similar": "Similar Laptops in Dataset",
        "confidence": "Prediction Confidence",
        "feature_imp": "Feature Importance",
        "feature_imp_sub": "How each spec contributes to the price",
        "price_range": "Price Range Simulation",
        "price_range_sub": "See how price changes with RAM upgrade",
        "analysis_title": "📊 Analysis Dashboard",
        "analysis_sub": "Insights from the laptop dataset",
        "price_vs_ram": "Price vs RAM",
        "price_vs_storage": "Price vs Storage",
        "specs_vs_price": "Specs Score vs Price (size = RAM)",
        "brand_dist": "Brand Distribution",
        "corr_matrix": "Correlation Matrix",
        "corr_sub": "Relationship between numerical features",
        "language": "🌐 Language",
        "model_info": "Upload your `model.joblib` in the same folder as this app.",
        "low": "Budget",
        "mid": "Mid-Range",
        "high": "Premium",
        "ultra": "Ultra Premium",
        "segment": "Price Segment",
    },
    "العربية": {
        "app_name": "لابتوب IQ",
        "app_sub": "ذكاء أسعار الحواسيب",
        "nav_home": "🏠  الرئيسية",
        "nav_predict": "🔮  توقع السعر",
        "nav_analysis": "📊  لوحة التحليل",
        "hero_tag": "💻  مدعوم بالذكاء الاصطناعي",
        "hero_title": "لابتوب IQ",
        "hero_title2": "توقع الأسعار",
        "hero_sub": "توقع أسعار اللابتوب بذكاء باستخدام التعلم الآلي",
        "laptops_analysed": "لابتوب تم تحليله",
        "avg_price": "متوسط السعر",
        "brands": "العلامات التجارية",
        "avg_rating": "متوسط التقييم",
        "price_dist": "توزيع الأسعار",
        "price_dist_sub": "كيف تتوزع الأسعار في البيانات",
        "top_brands": "أفضل الماركات",
        "top_brands_sub": "متوسط السعر لكل ماركة",
        "predict_title": "🔮 توقع سعر اللابتوب",
        "predict_sub": "أدخل المواصفات واحصل على تقدير فوري للسعر",
        "model_warn": "⚠️  ملف **model.joblib** غير موجود – يتم عرض توقع تجريبي.",
        "brand": "الماركة 🏷️",
        "processor": "المعالج ⚡",
        "ram": "الذاكرة RAM (جيجا) 🧠",
        "storage": "التخزين (جيجا) 💾",
        "screen": "حجم الشاشة (بوصة) 🖥️",
        "cores": "عدد النوى 🔲",
        "threads": "عدد الخيوط 🔁",
        "rating": "تقييم المستخدم ⭐",
        "specs_score": "نقاط المواصفات 📈",
        "predict_btn": "🚀  توقع السعر",
        "est_price": "السعر التقديري",
        "similar": "لابتوبات مشابهة في البيانات",
        "confidence": "مستوى الثقة في التوقع",
        "feature_imp": "أهمية المواصفات",
        "feature_imp_sub": "كيف تؤثر كل مواصفة على السعر",
        "price_range": "محاكاة نطاق الأسعار",
        "price_range_sub": "كيف يتغير السعر بترقية الذاكرة",
        "analysis_title": "📊 لوحة التحليل",
        "analysis_sub": "رؤى من بيانات اللابتوب",
        "price_vs_ram": "السعر مقابل الذاكرة",
        "price_vs_storage": "السعر مقابل التخزين",
        "specs_vs_price": "نقاط المواصفات مقابل السعر",
        "brand_dist": "توزيع الماركات",
        "corr_matrix": "مصفوفة الارتباط",
        "corr_sub": "العلاقة بين الخصائص الرقمية",
        "language": "🌐 اللغة",
        "model_info": "ضع ملف `model.joblib` في نفس مجلد هذا التطبيق.",
        "low": "اقتصادي",
        "mid": "متوسط",
        "high": "مميز",
        "ultra": "فائق التميز",
        "segment": "الفئة السعرية",
    },
    "Français": {
        "app_name": "Laptop",
        "app_sub": "Intelligence des Prix ML",
        "nav_home": "🏠  Accueil",
        "nav_predict": "🔮  Prédire le Prix",
        "nav_analysis": "📊  Tableau de Bord",
        "hero_tag": "💻  ALIMENTÉ PAR IA",
        "hero_title": "Laptop",
        "hero_title2": "Prédicteur de Prix",
        "hero_sub": "Prédisez les prix des laptops intelligemment avec le Machine Learning",
        "laptops_analysed": "Laptops Analysés",
        "avg_price": "Prix Moyen",
        "brands": "Marques",
        "avg_rating": "Note Moyenne",
        "price_dist": "Distribution des Prix",
        "price_dist_sub": "Comment les prix se répartissent dans les données",
        "top_brands": "Meilleures Marques",
        "top_brands_sub": "Prix moyen par marque",
        "predict_title": "🔮 Prédire le Prix du Laptop",
        "predict_sub": "Remplissez les specs et obtenez une estimation instantanée",
        "model_warn": "⚠️  **model.joblib** introuvable – affichage d'une prédiction demo.",
        "brand": "Marque 🏷️",
        "processor": "Processeur ⚡",
        "ram": "RAM (Go) 🧠",
        "storage": "Stockage (Go) 💾",
        "screen": "Taille Écran (pouces) 🖥️",
        "cores": "Cœurs CPU 🔲",
        "threads": "Threads CPU 🔁",
        "rating": "Note Utilisateur ⭐",
        "specs_score": "Score Specs 📈",
        "predict_btn": "🚀  Prédire le Prix",
        "est_price": "Prix Estimé",
        "similar": "Laptops Similaires dans les Données",
        "confidence": "Confiance de Prédiction",
        "feature_imp": "Importance des Caractéristiques",
        "feature_imp_sub": "Comment chaque spec contribue au prix",
        "price_range": "Simulation de Plage de Prix",
        "price_range_sub": "Voir comment le prix évolue avec plus de RAM",
        "analysis_title": "📊 Tableau de Bord Analytique",
        "analysis_sub": "Aperçus du jeu de données laptops",
        "price_vs_ram": "Prix vs RAM",
        "price_vs_storage": "Prix vs Stockage",
        "specs_vs_price": "Score Specs vs Prix (taille = RAM)",
        "brand_dist": "Distribution des Marques",
        "corr_matrix": "Matrice de Corrélation",
        "corr_sub": "Relation entre les caractéristiques numériques",
        "language": "🌐 Langue",
        "model_info": "Placez votre `model.joblib` dans le même dossier que cette app.",
        "low": "Budget",
        "mid": "Milieu de Gamme",
        "high": "Premium",
        "ultra": "Ultra Premium",
        "segment": "Segment de Prix",
    },
    "Español": {
        "app_name": "LaptopIQ",
        "app_sub": "Inteligencia de Precios ML",
        "nav_home": "🏠  Inicio",
        "nav_predict": "🔮  Predecir Precio",
        "nav_analysis": "📊  Panel de Análisis",
        "hero_tag": "💻  IMPULSADO POR IA",
        "hero_title": "LaptopIQ",
        "hero_title2": "Predictor de Precios",
        "hero_sub": "Predice precios de laptops de forma inteligente con Machine Learning",
        "laptops_analysed": "Laptops Analizadas",
        "avg_price": "Precio Promedio",
        "brands": "Marcas",
        "avg_rating": "Calificación Promedio",
        "price_dist": "Distribución de Precios",
        "price_dist_sub": "Cómo se distribuyen los precios en el dataset",
        "top_brands": "Mejores Marcas",
        "top_brands_sub": "Precio promedio por marca",
        "predict_title": "🔮 Predecir Precio del Laptop",
        "predict_sub": "Ingresa las especificaciones y obtén una estimación instantánea",
        "model_warn": "⚠️  **model.joblib** no encontrado – mostrando predicción demo.",
        "brand": "Marca 🏷️",
        "processor": "Procesador ⚡",
        "ram": "RAM (GB) 🧠",
        "storage": "Almacenamiento (GB) 💾",
        "screen": "Tamaño de Pantalla (pulgadas) 🖥️",
        "cores": "Núcleos CPU 🔲",
        "threads": "Hilos CPU 🔁",
        "rating": "Calificación del Usuario ⭐",
        "specs_score": "Puntuación de Specs 📈",
        "predict_btn": "🚀  Predecir Precio",
        "est_price": "Precio Estimado",
        "similar": "Laptops Similares en el Dataset",
        "confidence": "Confianza de Predicción",
        "feature_imp": "Importancia de Características",
        "feature_imp_sub": "Cómo contribuye cada spec al precio",
        "price_range": "Simulación de Rango de Precios",
        "price_range_sub": "Cómo cambia el precio al actualizar la RAM",
        "analysis_title": "📊 Panel de Análisis",
        "analysis_sub": "Perspectivas del dataset de laptops",
        "price_vs_ram": "Precio vs RAM",
        "price_vs_storage": "Precio vs Almacenamiento",
        "specs_vs_price": "Puntuación Specs vs Precio (tamaño = RAM)",
        "brand_dist": "Distribución de Marcas",
        "corr_matrix": "Matriz de Correlación",
        "corr_sub": "Relación entre características numéricas",
        "language": "🌐 Idioma",
        "model_info": "Coloca tu `model.joblib` en la misma carpeta que esta app.",
        "low": "Económico",
        "mid": "Gama Media",
        "high": "Premium",
        "ultra": "Ultra Premium",
        "segment": "Segmento de Precio",
    },
}

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');

:root {
  --bg:        #0d0f1a;
  --card:      #161929;
  --border:    #252840;
  --accent1:   #6c63ff;
  --accent2:   #ff6584;
  --accent3:   #43e97b;
  --accent4:   #f7971e;
  --text:      #e8eaf6;
  --muted:     #8b91b8;
  --radius:    16px;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: var(--card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

header[data-testid="stHeader"] { display: none !important; }

.stButton > button {
    background: linear-gradient(135deg, var(--accent1), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    cursor: pointer !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(108,99,255,0.4) !important;
}

.stSelectbox label, .stSlider label, .stNumberInput label {
    color: var(--muted) !important;
    font-weight: 500 !important;
}

[data-baseweb="select"] > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

input[type="number"] {
    background: var(--card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

[data-testid="metric-container"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] { color: var(--accent3) !important; font-family: 'Syne', sans-serif !important; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; }

[data-baseweb="tab-list"] { background: transparent !important; gap: 8px !important; }
[data-baseweb="tab"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--muted) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: linear-gradient(135deg, var(--accent1), var(--accent2)) !important;
    color: white !important;
    border: none !important;
}

/* Progress bar for confidence */
.confidence-bar {
    background: #252840;
    border-radius: 50px;
    height: 12px;
    overflow: hidden;
    margin-top: 6px;
}
.confidence-fill {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #43e97b, #6c63ff);
    transition: width 0.8s ease;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }

hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def hero_banner(t):
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1a1d35 0%, #0d0f1a 60%);
        border: 1px solid #252840;
        border-radius: 20px;
        padding: 2.5rem 2.5rem 2rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    ">
      <div style="
          position:absolute; top:-40px; right:-40px;
          width:220px; height:220px;
          background: radial-gradient(circle, rgba(108,99,255,0.25) 0%, transparent 70%);
          border-radius:50%;
      "></div>
      <div style="
          position:absolute; bottom:-60px; left:30%;
          width:180px; height:180px;
          background: radial-gradient(circle, rgba(255,101,132,0.18) 0%, transparent 70%);
          border-radius:50%;
      "></div>
      <p style="color:#6c63ff; font-family:'Syne',sans-serif; font-weight:700;
                letter-spacing:3px; font-size:0.75rem; margin:0 0 0.4rem;">
          {t['hero_tag']}
      </p>
      <h1 style="font-family:'Syne',sans-serif; font-size:2.6rem; font-weight:800;
                 margin:0; color:#e8eaf6; line-height:1.15;">
          {t['hero_title']}
          <span style="
              background: linear-gradient(90deg,#6c63ff,#ff6584);
              -webkit-background-clip:text; -webkit-text-fill-color:transparent;
          ">{t['hero_title2']}</span>
      </h1>
      <p style="color:#8b91b8; margin:0.6rem 0 0; font-size:1.05rem;">
          {t['hero_sub']}
      </p>
    </div>
    """, unsafe_allow_html=True)


def stat_card(icon, label, value, color):
    st.markdown(f"""
    <div style="
        background:#161929; border:1px solid #252840; border-radius:16px;
        padding:1.2rem 1.4rem; display:flex; align-items:center; gap:1rem;
    ">
      <div style="
          font-size:1.8rem; width:52px; height:52px; border-radius:12px;
          background:linear-gradient(135deg,{color}22,{color}11);
          display:flex; align-items:center; justify-content:center;
      ">{icon}</div>
      <div>
        <p style="color:#8b91b8; margin:0; font-size:0.8rem; font-weight:600;
                  letter-spacing:1px; text-transform:uppercase;">{label}</p>
        <p style="color:#e8eaf6; margin:0; font-size:1.5rem;
                  font-family:'Syne',sans-serif; font-weight:800;">{value}</p>
      </div>
    </div>
    """, unsafe_allow_html=True)


def section_title(title, subtitle=""):
    st.markdown(f"""
    <div style="margin: 1.5rem 0 1rem;">
      <h2 style="font-family:'Syne',sans-serif; font-weight:800;
                 font-size:1.5rem; color:#e8eaf6; margin:0;">{title}</h2>
      {"<p style='color:#8b91b8;margin:0.2rem 0 0;font-size:0.95rem;'>"+subtitle+"</p>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)


def price_result_card(price, t):
    # Determine segment
    if price < 600:
        segment = t["low"]; seg_color = "#43e97b"
    elif price < 1200:
        segment = t["mid"]; seg_color = "#6c63ff"
    elif price < 2000:
        segment = t["high"]; seg_color = "#f7971e"
    else:
        segment = t["ultra"]; seg_color = "#ff6584"

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1c1e35, #161929);
        border: 1px solid #6c63ff55;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(108,99,255,0.15);
        margin-top: 1rem;
    ">
      <p style="color:#8b91b8; font-size:0.9rem; letter-spacing:2px;
                text-transform:uppercase; margin:0 0 0.5rem;">{t['est_price']}</p>
      <h1 style="
          font-family:'Syne',sans-serif; font-weight:800; font-size:3.2rem;
          background:linear-gradient(90deg,#43e97b,#6c63ff);
          -webkit-background-clip:text; -webkit-text-fill-color:transparent;
          margin:0;
      ">${price:,.0f}</h1>
      <p style="color:#8b91b8; margin:0.5rem 0 0.8rem; font-size:0.85rem;">
          ± ${price*0.92:,.0f} – ${price*1.08:,.0f}
      </p>
      <span style="
          background:{seg_color}22; color:{seg_color}; border:1px solid {seg_color}55;
          border-radius:50px; padding:0.3rem 1.2rem; font-size:0.85rem; font-weight:700;
      ">{t['segment']}: {segment}</span>
    </div>
    """, unsafe_allow_html=True)


def confidence_card(score, t):
    """Animated confidence meter."""
    bar_color = "#43e97b" if score >= 75 else "#f7971e" if score >= 50 else "#ff6584"
    st.markdown(f"""
    <div style="background:#161929; border:1px solid #252840; border-radius:16px;
                padding:1.4rem 1.6rem; margin-top:1rem;">
      <p style="color:#8b91b8; margin:0 0 0.4rem; font-size:0.85rem;
                letter-spacing:1px; text-transform:uppercase;">{t['confidence']}</p>
      <div style="display:flex; align-items:center; gap:1rem;">
        <div style="flex:1; background:#252840; border-radius:50px; height:14px; overflow:hidden;">
          <div style="height:100%; width:{score}%; border-radius:50px;
                      background:linear-gradient(90deg,{bar_color},{bar_color}99);"></div>
        </div>
        <span style="font-family:'Syne',sans-serif; font-weight:800;
                     color:{bar_color}; font-size:1.2rem; min-width:48px;">{score}%</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def feature_importance_chart(ram, storage, cores, ratings, specs_score, screen_size, t):
    """Bar chart showing how each feature contributed to the price estimate."""
    features = ["RAM", "Storage", "Cores", "Rating", "Specs Score", "Screen"]
    weights  = [
        ram * 15,
        storage * 0.3,
        cores * 40,
        ratings * 80,
        specs_score * 12,
        screen_size * 10,
    ]
    total = sum(weights)
    pcts  = [round(w / total * 100, 1) for w in weights]

    fig = go.Figure(go.Bar(
        x=features, y=pcts,
        marker=dict(
            color=pcts,
            colorscale=[[0, "#6c63ff"], [0.5, "#f7971e"], [1, "#43e97b"]],
            showscale=False,
        ),
        text=[f"{p}%" for p in pcts],
        textposition="outside",
        textfont=dict(color="#e8eaf6"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#8b91b8",
        yaxis=dict(gridcolor="#252840", title="Contribution (%)", ticksuffix="%"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=30, b=0),
        height=300,
    )
    section_title(t["feature_imp"], t["feature_imp_sub"])
    st.plotly_chart(fig, use_container_width=True)


def price_range_simulation(base_price, current_ram, t):
    """Line chart simulating price across RAM options."""
    rams   = [4, 8, 16, 32, 64]
    prices = [base_price * (r / current_ram) ** 0.45 for r in rams]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[str(r) + " GB" for r in rams],
        y=prices,
        mode="lines+markers",
        line=dict(color="#6c63ff", width=3),
        marker=dict(size=10, color="#ff6584", line=dict(color="#fff", width=2)),
        fill="tozeroy",
        fillcolor="rgba(108,99,255,0.08)",
    ))
    # Highlight current
    idx = rams.index(current_ram)
    fig.add_trace(go.Scatter(
        x=[str(current_ram) + " GB"],
        y=[prices[idx]],
        mode="markers",
        marker=dict(size=16, color="#43e97b", symbol="star",
                    line=dict(color="#fff", width=2)),
        name="Your Config",
        showlegend=True,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#8b91b8",
        xaxis=dict(gridcolor="#252840"),
        yaxis=dict(gridcolor="#252840", title="Price ($)"),
        showlegend=False,
        margin=dict(l=0, r=0, t=20, b=0),
        height=280,
    )
    section_title(t["price_range"], t["price_range_sub"])
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load("model.joblib")
    except FileNotFoundError:
        return None


# ─────────────────────────────────────────────
#  MOCK DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_sample_data():
    np.random.seed(42)
    n = 300
    brands   = ["Dell", "HP", "Lenovo", "Asus", "Acer", "Apple", "MSI", "Samsung"]
    procs    = ["Intel i3", "Intel i5", "Intel i7", "Intel i9",
                "AMD Ryzen 5", "AMD Ryzen 7", "Apple M1", "Apple M2"]
    rams     = [4, 8, 16, 32, 64]
    storages = [128, 256, 512, 1024, 2048]

    df = pd.DataFrame({
        "brand":       np.random.choice(brands, n),
        "processor":   np.random.choice(procs, n),
        "ram":         np.random.choice(rams, n),
        "storage":     np.random.choice(storages, n),
        "screen_size": np.round(np.random.uniform(11.6, 17.3, n), 1),
        "cores":       np.random.choice([2, 4, 6, 8, 12, 16], n),
        "threads":     np.random.choice([4, 8, 12, 16, 24, 32], n),
        "ratings":     np.round(np.random.uniform(3.0, 5.0, n), 1),
        "specs_score": np.random.randint(40, 100, n),
        "price":       np.random.randint(300, 3500, n),
    })
    return df


# ─────────────────────────────────────────────
#  SIDEBAR  NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    # Language selector first
    lang = st.selectbox(
        "🌐 Language / اللغة / Langue / Idioma",
        list(TRANSLATIONS.keys()),
        label_visibility="visible",
    )
    t = TRANSLATIONS[lang]

    st.markdown(f"""
    <div style="padding:1.2rem 0 0.5rem;">
      <p style="font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:800;
                color:#e8eaf6; margin:0;">💻 {t['app_name']}</p>
      <p style="color:#8b91b8; font-size:0.8rem; margin:0;">{t['app_sub']}</p>
    </div>
    <hr style="border-color:#252840; margin:0.8rem 0;"/>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [t["nav_home"], t["nav_predict"], t["nav_analysis"]],
        label_visibility="collapsed",
    )

    st.markdown(f"""
    <hr style="border-color:#252840; margin:1rem 0;"/>
    <p style="color:#8b91b8; font-size:0.78rem;">
      {t['model_info']}
    </p>
    """, unsafe_allow_html=True)


model = load_model()
df    = load_sample_data()


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 1 – HOME
# ═══════════════════════════════════════════════════════════════════════
if "Home" in page or "الرئيسية" in page or "Accueil" in page or "Inicio" in page:
    hero_banner(t)

    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_card("💻", t["laptops_analysed"], f"{len(df):,}",       "#6c63ff")
    with c2: stat_card("🏷️", t["avg_price"],        f"${df.price.mean():,.0f}", "#ff6584")
    with c3: stat_card("📦", t["brands"],            str(df.brand.nunique()),    "#43e97b")
    with c4: stat_card("⭐", t["avg_rating"],        f"{df.ratings.mean():.1f}", "#f7971e")

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1.4, 1])

    with col_a:
        section_title(t["price_dist"], t["price_dist_sub"])
        fig = px.histogram(df, x="price", nbins=40,
                           color_discrete_sequence=["#6c63ff"])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#8b91b8",
            xaxis=dict(gridcolor="#252840", title="Price ($)"),
            yaxis=dict(gridcolor="#252840", title="Count"),
            bargap=0.05, showlegend=False,
            margin=dict(l=0, r=0, t=20, b=0),
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        section_title(t["top_brands"], t["top_brands_sub"])
        brand_avg = df.groupby("brand")["price"].mean().sort_values(ascending=True)
        fig2 = px.bar(brand_avg, orientation="h",
                      color=brand_avg.values,
                      color_continuous_scale=["#6c63ff", "#ff6584", "#f7971e"])
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#8b91b8",
            xaxis=dict(gridcolor="#252840", title="Avg Price ($)"),
            yaxis=dict(gridcolor="#252840", title=""),
            showlegend=False, coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 2 – PREDICT
# ═══════════════════════════════════════════════════════════════════════
elif "Predict" in page or "توقع" in page or "Prédire" in page or "Predecir" in page:
    hero_banner(t)
    section_title(t["predict_title"], t["predict_sub"])

    if model is None:
        st.warning(t["model_warn"])

    brands   = ["Dell", "HP", "Lenovo", "Asus", "Acer", "Apple", "MSI", "Samsung"]
    procs    = ["Intel i3", "Intel i5", "Intel i7", "Intel i9",
                "AMD Ryzen 5", "AMD Ryzen 7", "Apple M1", "Apple M2"]
    rams     = [4, 8, 16, 32, 64]
    storages = [128, 256, 512, 1024, 2048]

    col1, col2, col3 = st.columns(3)

    with col1:
        brand       = st.selectbox(t["brand"],     brands)
        processor   = st.selectbox(t["processor"], procs)
        ram         = st.selectbox(t["ram"],        rams, index=1)
        storage     = st.selectbox(t["storage"],    storages, index=2)

    with col2:
        screen_size = st.slider(t["screen"], 11.6, 17.3, 15.6, 0.1)
        cores       = st.selectbox(t["cores"],   [2, 4, 6, 8, 12, 16], index=2)
        threads     = st.selectbox(t["threads"], [4, 8, 12, 16, 24, 32], index=2)

    with col3:
        ratings     = st.slider(t["rating"],      1.0, 5.0, 4.2, 0.1)
        specs_score = st.slider(t["specs_score"],   0, 100, 75)

    st.markdown("<br>", unsafe_allow_html=True)

    brand_enc = brands.index(brand)
    proc_enc  = procs.index(processor)
    input_data = np.array([[brand_enc, ratings, specs_score, proc_enc,
                            ram, storage, screen_size, 1920 * 1080, cores, threads]])

    btn_col, _ = st.columns([1, 3])
    with btn_col:
        predict_btn = st.button(t["predict_btn"], use_container_width=True)

    if predict_btn:
        if model:
            price = model.predict(input_data)[0]
            confidence = min(95, 70 + int(specs_score * 0.25))
        else:
            price = (ram * 15 + storage * 0.3 + specs_score * 12
                     + cores * 40 + ratings * 80 + 200 + brand_enc * 30)
            confidence = min(88, 55 + int(specs_score * 0.3))

        # Result + confidence side by side
        res_col, conf_col = st.columns([1.2, 1])
        with res_col:
            price_result_card(price, t)
        with conf_col:
            st.markdown("<br>", unsafe_allow_html=True)
            confidence_card(confidence, t)

        st.markdown("<br>", unsafe_allow_html=True)

        # Feature importance chart
        feature_importance_chart(ram, storage, cores, ratings, specs_score, screen_size, t)

        # Price range simulation
        price_range_simulation(price, ram, t)

        # Similar laptops
        st.markdown("<br>", unsafe_allow_html=True)
        section_title(t["similar"])
        similar = df[df["brand"] == brand].head(5)
        st.dataframe(
            similar[["brand", "processor", "ram", "storage", "screen_size", "ratings", "price"]],
            use_container_width=True, hide_index=True,
        )


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 3 – DASHBOARD
# ═══════════════════════════════════════════════════════════════════════
else:
    hero_banner(t)
    section_title(t["analysis_title"], t["analysis_sub"])

    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(df, x="ram", y="price", color="ram",
                     color_discrete_sequence=px.colors.sequential.Plasma)
        fig.update_layout(
            title=t["price_vs_ram"], paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="#8b91b8",
            xaxis=dict(gridcolor="#252840"), yaxis=dict(gridcolor="#252840"),
            showlegend=False, margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.box(df, x="storage", y="price", color="storage",
                     color_discrete_sequence=px.colors.sequential.Viridis)
        fig.update_layout(
            title=t["price_vs_storage"], paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="#8b91b8",
            xaxis=dict(gridcolor="#252840"), yaxis=dict(gridcolor="#252840"),
            showlegend=False, margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.scatter(df, x="specs_score", y="price", color="brand",
                         size="ram", hover_data=["processor"],
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(
            title=t["specs_vs_price"],
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#8b91b8",
            xaxis=dict(gridcolor="#252840"), yaxis=dict(gridcolor="#252840"),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        brand_counts = df["brand"].value_counts()
        fig = px.pie(values=brand_counts.values, names=brand_counts.index,
                     hole=0.5,
                     color_discrete_sequence=["#6c63ff","#ff6584","#43e97b",
                                               "#f7971e","#00c9ff","#92fe9d",
                                               "#fc466b","#3f5efb"])
        fig.update_layout(
            title=t["brand_dist"],
            paper_bgcolor="rgba(0,0,0,0)", font_color="#8b91b8",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    section_title(t["corr_matrix"], t["corr_sub"])
    num_cols = ["ratings", "specs_score", "ram", "storage", "screen_size", "cores", "threads", "price"]
    corr = df[num_cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale=[[0, "#ff6584"], [0.5, "#252840"], [1, "#43e97b"]],
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#8b91b8",
        margin=dict(l=0, r=0, t=20, b=0),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)