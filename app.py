import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered",
    initial_sidebar_state="collapsed",
)

TRANSLATIONS = {
    "English": {
        "title": "Laptop Price Predictor",
        "subtitle": "Enter the specifications to get an estimated price in MAD",
        "brand": "Brand",
        "processor": "Processor",
        "ram": "RAM (GB)",
        "storage": "Storage (GB)",
        "cores": "Cores",
        "threads": "Threads",
        "specs_score": "Specs Score",
        "predict_btn": "🔮 Predict Price",
        "est_price": "Estimated Price",
        "low": "Budget",
        "mid": "Mid-Range",
        "high": "Premium",
        "ultra": "Ultra Premium",
        "segment": "Segment",
        "confidence": "Confidence",
        "similar": "Similar Laptops",
        "feature_imp": "Feature Importance",
        "price_range": "Price Simulation",
        "analysis_title": "Analysis Dashboard",
        "price_vs_ram": "Price vs RAM",
        "price_vs_storage": "Price vs Storage",
        "specs_vs_price": "Specs vs Price",
        "brand_dist": "Brand Distribution",
        "corr_matrix": "Correlation Matrix",
        "nav_predict": "Predict",
        "nav_analysis": "Analysis",
        "nav_home": "Home",
        "laptops_analysed": "Laptops",
        "avg_price": "Avg Price",
        "brands": "Brands",
        "avg_rating": "Avg Rating",
        "price_dist": "Price Distribution",
        "top_brands": "Top Brands",
        "model_warn": "model.joblib not found - showing demo prediction",
        "language": "Language",
    },
    "Français": {
        "title": "Prédicteur de Prix Laptop",
        "subtitle": "Entrez les spécifications pour obtenir un prix estimé en MAD",
        "brand": "Marque",
        "processor": "Processeur",
        "ram": "RAM (Go)",
        "storage": "Stockage (Go)",
        "cores": "Cœurs",
        "threads": "Threads",
        "specs_score": "Score Specs",
        "predict_btn": "🔮 Prédire le Prix",
        "est_price": "Prix Estimé",
        "low": "Budget",
        "mid": "Milieu de Gamme",
        "high": "Premium",
        "ultra": "Ultra Premium",
        "segment": "Segment",
        "confidence": "Confiance",
        "similar": "Laptops Similaires",
        "feature_imp": "Importance",
        "price_range": "Simulation Prix",
        "analysis_title": "Tableau de Bord",
        "price_vs_ram": "Prix vs RAM",
        "price_vs_storage": "Prix vs Stockage",
        "specs_vs_price": "Specs vs Prix",
        "brand_dist": "Distribution Marques",
        "corr_matrix": "Matrice Corrélation",
        "nav_predict": "Prédire",
        "nav_analysis": "Analyse",
        "nav_home": "Accueil",
        "laptops_analysed": "Laptops",
        "avg_price": "Prix Moyen",
        "brands": "Marques",
        "avg_rating": "Note Moyenne",
        "price_dist": "Distribution Prix",
        "top_brands": "Meilleures Marques",
        "model_warn": "model.joblib introuvable - prédiction demo",
        "language": "Langue",
    },
    "العربية": {
        "title": "متنبئ أسعار اللابتوب",
        "subtitle": "أدخل المواصفات للحصول على سعر تقديري بالدرهم",
        "brand": "الماركة",
        "processor": "المعالج",
        "ram": "الذاكرة (جيجا)",
        "storage": "التخزين (جيجا)",
        "cores": "النوى",
        "threads": "الخيوط",
        "specs_score": "نقاط المواصفات",
        "predict_btn": "🔮 توقع السعر",
        "est_price": "السعر التقديري",
        "low": "اقتصادي",
        "mid": "متوسط",
        "high": "مميز",
        "ultra": "فائق التميز",
        "segment": "الفئة",
        "confidence": "الثقة",
        "similar": "لابتوبات مشابهة",
        "feature_imp": "أهمية المواصفات",
        "price_range": "محاكاة السعر",
        "analysis_title": "لوحة التحليل",
        "price_vs_ram": "السعر مقابل الذاكرة",
        "price_vs_storage": "السعر مقابل التخزين",
        "specs_vs_price": "المواصفات مقابل السعر",
        "brand_dist": "توزيع الماركات",
        "corr_matrix": "مصفوفة الارتباط",
        "nav_predict": "توقع",
        "nav_analysis": "تحليل",
        "nav_home": "الرئيسية",
        "laptops_analysed": "لابتوب",
        "avg_price": "متوسط السعر",
        "brands": "الماركات",
        "avg_rating": "متوسط التقييم",
        "price_dist": "توزيع الأسعار",
        "top_brands": "أفضل الماركات",
        "model_warn": "ملف model.joblib غير موجود - توقع تجريبي",
        "language": "اللغة",
    },
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #0f0c29;
  --bg-gradient: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
  --card: rgba(255,255,255,0.05);
  --card-border: rgba(255,255,255,0.1);
  --text: #ffffff;
  --text-muted: #a0a0c0;
  --accent1: #6366f1;
  --accent2: #ec4899;
  --accent3: #10b981;
  --accent4: #f59e0b;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    background-image: var(--bg-gradient) !important;
    color: var(--text) !important;
    font-family: 'Poppins', sans-serif !important;
    min-height: 100vh;
}

[data-testid="stHeader"] { display: none !important; }

/* Main container centered */
.block-container {
    max-width: 800px !important;
    padding: 2rem 1rem !important;
}

/* Title styling */
.title-gradient {
    background: linear-gradient(90deg, #6366f1, #ec4899, #f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    font-size: 2.5rem;
    text-align: center;
    margin: 0;
    letter-spacing: -1px;
}

.subtitle {
    color: var(--text-muted);
    text-align: center;
    font-size: 0.95rem;
    margin: 0.5rem 0 2rem;
}

/* Form cards */
.form-card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 20px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1rem;
}

/* Select boxes */
[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
}

/* Slider */
[data-testid="stSlider"] > div > div > div {
    background: rgba(255,255,255,0.1) !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: linear-gradient(90deg, #6366f1, #ec4899) !important;
    border: 2px solid white !important;
    box-shadow: 0 0 10px rgba(99,102,241,0.5) !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #6366f1, #ec4899) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.9rem 3rem !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 8px 30px rgba(99,102,241,0.4) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 40px rgba(99,102,241,0.6) !important;
}

/* Labels */
.stSelectbox label, .stSlider label {
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(236,72,153,0.15));
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 24px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
    backdrop-filter: blur(20px);
}

/* Navigation */
.nav-container {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-bottom: 2rem;
}
.nav-btn {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px;
    padding: 0.6rem 1.5rem;
    color: var(--text-muted);
    text-decoration: none;
    font-weight: 500;
    transition: all 0.3s ease;
    cursor: pointer;
}
.nav-btn:hover, .nav-btn.active {
    background: rgba(99,102,241,0.2);
    border-color: rgba(99,102,241,0.5);
    color: white;
}

/* Language selector */
.lang-select {
    position: absolute;
    top: 1rem;
    right: 1rem;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        return joblib.load("model.joblib")
    except FileNotFoundError:
        return None


@st.cache_data
def load_sample_data():
    np.random.seed(42)
    n = 300
    brands   = ["Dell", "HP", "Lenovo", "Asus", "Acer", "Apple", "MSI", "Samsung"]
    procs    = ["Intel Core i3", "Intel Core i5", "Intel Core i7", "Intel Core i9",
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


model = load_model()
df    = load_sample_data()


# Language selector at top
lang_col, _ = st.columns([1, 3])
with lang_col:
    lang = st.selectbox(
        "🌐",
        list(TRANSLATIONS.keys()),
        label_visibility="collapsed",
    )
t = TRANSLATIONS[lang]


# Navigation
st.markdown(f"""
<div style="display:flex; justify-content:center; gap:0.8rem; margin-bottom:2rem;">
    <a href="?page=home" style="background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.15);
       border-radius:12px; padding:0.5rem 1.2rem; color:#a0a0c0; text-decoration:none;
       font-weight:500; font-size:0.9rem;">{t['nav_home']}</a>
    <a href="?page=predict" style="background:rgba(99,102,241,0.2); border:1px solid rgba(99,102,241,0.5);
       border-radius:12px; padding:0.5rem 1.2rem; color:white; text-decoration:none;
       font-weight:600; font-size:0.9rem;">{t['nav_predict']}</a>
    <a href="?page=analysis" style="background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.15);
       border-radius:12px; padding:0.5rem 1.2rem; color:#a0a0c0; text-decoration:none;
       font-weight:500; font-size:0.9rem;">{t['nav_analysis']}</a>
</div>
""", unsafe_allow_html=True)


# Title
st.markdown(f'<h1 class="title-gradient">💻 {t["title"]}</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitle">{t["subtitle"]}</p>', unsafe_allow_html=True)


# Get page from query params
query_params = st.query_params
page = query_params.get("page", "predict")


if page == "predict":
    if model is None:
        st.warning(t["model_warn"])

    brands   = ["Dell", "HP", "Lenovo", "Asus", "Acer", "Apple", "MSI", "Samsung"]
    procs    = ["Intel Core i3", "Intel Core i5", "Intel Core i7", "Intel Core i9",
                "AMD Ryzen 5", "AMD Ryzen 7", "Apple M1", "Apple M2"]
    rams     = [4, 8, 16, 32, 64]
    storages = [128, 256, 512, 1024, 2048]

    st.markdown('<div class="form-card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        brand       = st.selectbox(t["brand"],     brands)
        ram         = st.selectbox(t["ram"],        rams, index=1)
        cores       = st.selectbox(t["cores"],   [2, 4, 6, 8, 12, 16], index=1)
        specs_score = st.slider(t["specs_score"], 0, 100, 50)

    with col2:
        processor   = st.selectbox(t["processor"], procs)
        storage     = st.selectbox(t["storage"],    storages, index=1)
        threads     = st.selectbox(t["threads"], [4, 8, 12, 16, 24, 32], index=1)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    btn_col, _ = st.columns([2, 1])
    with btn_col:
        predict_btn = st.button(t["predict_btn"], use_container_width=True)

    if predict_btn:
        brand_enc = brands.index(brand)
        proc_enc  = procs.index(processor)

        if model:
            input_data = np.array([[brand_enc, 4.0, specs_score, proc_enc,
                                    ram, storage, 15.6, 1920 * 1080, cores, threads]])
            price = model.predict(input_data)[0]
            confidence = min(95, 70 + int(specs_score * 0.25))
        else:
            price = (ram * 15 + storage * 0.3 + specs_score * 12
                     + cores * 40 + 4.0 * 80 + 200 + brand_enc * 30)
            confidence = min(88, 55 + int(specs_score * 0.3))

        if price < 600:
            segment = t["low"]; seg_color = "#10B981"
        elif price < 1200:
            segment = t["mid"]; seg_color = "#6366F1"
        elif price < 2000:
            segment = t["high"]; seg_color = "#F59E0B"
        else:
            segment = t["ultra"]; seg_color = "#EC4899"

        st.markdown(f"""
        <div class="result-card">
          <p style="color:#a0a0c0; font-size:0.85rem; letter-spacing:2px; text-transform:uppercase; margin:0 0 0.5rem;">{t['est_price']}</p>
          <h1 style="font-family:'Poppins',sans-serif; font-weight:800; font-size:3rem; color:white; margin:0;">{price:,.0f} MAD</h1>
          <p style="color:#a0a0c0; margin:0.5rem 0 0.8rem; font-size:0.9rem;">Range: {price*0.88:,.0f} - {price*1.12:,.0f} MAD</p>
          <span style="background:{seg_color}22; color:{seg_color}; border:1px solid {seg_color}50; border-radius:50px; padding:0.3rem 1rem; font-size:0.8rem; font-weight:600;">{t['segment']}: {segment}</span>
        </div>
        """, unsafe_allow_html=True)

        bar_color = "#10B981" if confidence >= 75 else "#F59E0B" if confidence >= 50 else "#EF4444"
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); border-radius:16px; padding:1.2rem; margin-top:1rem;">
          <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem;">
            <span style="color:#a0a0c0; font-size:0.8rem;">{t['confidence']}</span>
            <span style="color:{bar_color}; font-weight:700;">{confidence}%</span>
          </div>
          <div style="background:rgba(255,255,255,0.1); border-radius:50px; height:8px;">
            <div style="width:{confidence}%; height:100%; border-radius:50px; background:linear-gradient(90deg,{bar_color},{bar_color}80);"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)


elif page == "analysis":
    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 " + t["price_vs_ram"], "🔧 " + t["specs_vs_price"], "📊 " + t["brand_dist"]])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(df, x="ram", y="price", color="ram", color_discrete_sequence=px.colors.sequential.Plasma)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#a0a0c0", showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.box(df, x="storage", y="price", color="storage", color_discrete_sequence=px.colors.sequential.Viridis)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#a0a0c0", showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(df, x="specs_score", y="price", color="brand", size="ram", color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#a0a0c0", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            num_cols = ["ratings", "specs_score", "ram", "storage", "screen_size", "cores", "threads", "price"]
            corr = df[num_cols].corr()
            fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale=[[0, "#EC4899"], [0.5, "#1F2937"], [1, "#10B981"]], zmid=0, text=np.round(corr.values, 2), texttemplate="%{text}"))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#a0a0c0", margin=dict(l=0, r=0, t=20, b=0), height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        brand_counts = df["brand"].value_counts()
        fig = px.pie(values=brand_counts.values, names=brand_counts.index, hole=0.55, color_discrete_sequence=["#6366F1","#EC4899","#10B981","#F59E0B","#06B6D4","#8B5CF6","#F43F5E","#3B82F6"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#a0a0c0", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)


else:  # Home page
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("💻", t["laptops_analysed"], f"{len(df):,}", "#6366F1"),
        ("💰", t["avg_price"], f"{df.price.mean():,.0f} MAD", "#EC4899"),
        ("🏷️", t["brands"], str(df.brand.nunique()), "#10B981"),
        ("⭐", t["avg_rating"], f"{df.ratings.mean():.1f}", "#F59E0B"),
    ]
    for col, (icon, label, value, color) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); border-radius:16px; padding:1.2rem; text-align:center;">
              <div style="font-size:1.5rem; margin-bottom:0.3rem;">{icon}</div>
              <div style="color:#a0a0c0; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px;">{label}</div>
              <div style="color:white; font-size:1.3rem; font-weight:700; margin-top:0.2rem;">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1.4, 1])
    with col_a:
        fig = px.histogram(df, x="price", nbins=40, color_discrete_sequence=["#6366F1"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#a0a0c0", showlegend=False, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        brand_avg = df.groupby("brand")["price"].mean().sort_values(ascending=True)
        fig = px.bar(brand_avg, orientation="h", color=brand_avg.values, color_continuous_scale=["#6366F1", "#EC4899", "#F59E0B"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#a0a0c0", showlegend=False, coloraxis_showscale=False, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)