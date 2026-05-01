import streamlit as st
import numpy as np
import joblib
import os
model_path = "best_xgb_model.pkl"
if not os.path.exists(model_path):
    st.error(f"⚠️ Model file '{model_path}' not found. Please make sure it's in the same folder as this app.")
else:
    model = joblib.load(model_path)

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered"
)

# ─── Translations ──────────────────────────────────────────────────────────────
translations = {
    "English": {
        "title": "💻 Laptop Price Predictor",
        "subtitle": "Enter the specifications to get an estimated price in MAD",
        "brand": "Brand",
        "processor": "Processor",
        "ram": "RAM (GB)",
        "storage": "Storage (GB)",
        "cores": "Cores",
        "threads": "Threads",
        "specs_score": "Specs Score",
        "predict": "🔍 Predict Price",
        "result": "Estimated Price",
        "currency": "MAD",
        "select": "Select...",
    },
    "Français": {
        "title": "💻 Prédicteur de Prix Laptop",
        "subtitle": "Entrez les spécifications pour obtenir un prix estimé en MAD",
        "brand": "Marque",
        "processor": "Processeur",
        "ram": "RAM (Go)",
        "storage": "Stockage (Go)",
        "cores": "Cœurs",
        "threads": "Threads",
        "specs_score": "Score Specs",
        "predict": "🔍 Prédire le Prix",
        "result": "Prix Estimé",
        "currency": "MAD",
        "select": "Choisir...",
    },
    "العربية": {
        "title": "💻 توقع سعر اللاپتوب",
        "subtitle": "أدخل المواصفات للحصول على السعر التقديري بالدرهم",
        "brand": "الماركة",
        "processor": "المعالج",
        "ram": "الذاكرة العشوائية (GB)",
        "storage": "التخزين (GB)",
        "cores": "النوى",
        "threads": "الخيوط",
        "specs_score": "نقاط المواصفات",
        "predict": "🔍 توقع السعر",
        "result": "السعر التقديري",
        "currency": "درهم",
        "select": "اختر...",
    }
}

# ─── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #0f0f1a 100%);
    min-height: 100vh;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00d4ff, #7b2fff, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.2rem;
}

.subtitle {
    text-align: center;
    color: #8888aa;
    font-size: 1rem;
    margin-bottom: 2rem;
    font-weight: 300;
}

.lang-label {
    text-align: center;
    color: #aaaacc;
    font-size: 0.85rem;
    margin-bottom: 0.3rem;
    font-family: 'Syne', sans-serif;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.result-box {
    background: linear-gradient(135deg, #1e1e3a, #2a1a4a);
    border: 1px solid #7b2fff44;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
    box-shadow: 0 0 40px #7b2fff22;
}

.result-label {
    color: #aaaacc;
    font-size: 0.9rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-family: 'Syne', sans-serif;
    margin-bottom: 0.5rem;
}

.result-price {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00d4ff, #7b2fff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.result-currency {
    color: #aaaacc;
    font-size: 1.2rem;
    margin-top: 0.3rem;
}

.card {
    background: #1a1a2e;
    border: 1px solid #ffffff11;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label {
    color: #ccccee !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
}

div[data-testid="stSelectbox"] > div > div {
    background: #1e1e3a !important;
    border: 1px solid #7b2fff44 !important;
    color: white !important;
    border-radius: 8px !important;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #7b2fff, #00d4ff) !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    padding: 0.75rem !important;
    border: none !important;
    border-radius: 10px !important;
    letter-spacing: 0.05em !important;
    transition: all 0.3s ease !important;
    margin-top: 1rem;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px #7b2fff55 !important;
}

.divider {
    border: none;
    border-top: 1px solid #ffffff11;
    margin: 1.5rem 0;
}

.stApp > header {display: none;}
</style>
""", unsafe_allow_html=True)

# ─── Language Selector ─────────────────────────────────────────────────────────
st.markdown('<div class="lang-label">🌐 Language / Langue / اللغة</div>', unsafe_allow_html=True)
lang = st.selectbox("", ["English", "Français", "العربية"], label_visibility="collapsed")
t = translations[lang]

# ─── Title ─────────────────────────────────────────────────────────────────────
st.markdown(f'<div class="main-title">{t["title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtitle">{t["subtitle"]}</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─── Brand & Processor Maps ────────────────────────────────────────────────────
brand_map = {
    "HP": 5, "Lenovo": 4, "Asus": 3, "Acer": 2,
    "MSI": 6, "Dell": 1, "Samsung": 7, "Infinix": 0
}

processor_map = {
    "Intel Core i3": 3, "Intel Core i5": 5, "Intel Core i7": 7,
    "Intel Core i9": 9, "Intel Core Ultra 5": 10, "Intel Core Ultra 7": 11,
    "Intel Core Ultra 9": 12, "AMD Ryzen 3": 1, "AMD Ryzen 5": 6,
    "AMD Ryzen 7": 8, "AMD Ryzen 9": 13, "Apple M1": 14,
    "Snapdragon": 210, "MediaTek": 390
}

# ─── Input Form ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    brand_name = st.selectbox(f"🏷️ {t['brand']}", list(brand_map.keys()))
    ram = st.selectbox(f"🧠 {t['ram']}", [8, 12, 16, 24, 32])
    cores = st.selectbox(f"⚙️ {t['cores']}", [2, 4, 5, 10, 12, 14, 16, 20, 24])
    specs_score = st.slider(f"📊 {t['specs_score']}", min_value=0, max_value=100, value=50, step=1)

with col2:
    processor_name = st.selectbox(f"🔧 {t['processor']}", list(processor_map.keys()))
    storage = st.selectbox(f"💾 {t['storage']}", [256, 512, 1024])
    threads = st.selectbox(f"🔀 {t['threads']}", [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32])

# ─── Predict ───────────────────────────────────────────────────────────────────
if st.button(t["predict"]):
    try:
        model = joblib.load("best_xgb_model.pkl")

        input_data = np.array([[
            specs_score,
            brand_map[brand_name],
            processor_map[processor_name],
            ram,
            storage,
            cores,
            threads,
        ]])

        # Feature order must match training:
        # specs_score, screen_size, threads, ram, cores, brand, ratings, processor, resolution, storage
        # Adjust below to match your exact X.columns order!
        input_data = np.array([[
            specs_score,   # specs_score
            13.3,          # screen_size (default)
            threads,       # threads
            ram,           # ram
            cores,         # cores
            brand_map[brand_name],  # brand
            4.2,           # ratings (default)
            processor_map[processor_name],  # processor
            1920,          # resolution (default)
            storage,       # storage
        ]])

        price = model.predict(input_data)[0]
        price = max(0, price)

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">{t['result']}</div>
            <div class="result-price">{price:,.0f}</div>
            <div class="result-currency">{t['currency']}</div>
        </div>
        """, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("⚠️ Model file 'best_xgb_model.pkl' not found. Please make sure it's in the same folder.")
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")