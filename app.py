import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
 
# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LaptopIQ – Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ─────────────────────────────────────────────
#  GLOBAL CSS  – Colorful & Modern
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
 
/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
 
/* Hide default header */
header[data-testid="stHeader"] { display: none !important; }
 
/* Buttons */
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
 
/* Selectbox / Slider labels */
.stSelectbox label, .stSlider label, .stNumberInput label {
    color: var(--muted) !important;
    font-weight: 500 !important;
}
 
/* Selectbox input */
[data-baseweb="select"] > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
 
/* Number input */
input[type="number"] {
    background: var(--card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
 
/* Metric cards */
[data-testid="metric-container"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] { color: var(--accent3) !important; font-family: 'Syne', sans-serif !important; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; }
 
/* Tabs */
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
 
/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
 
/* Divider */
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────
#  HELPERS  – HTML components
# ─────────────────────────────────────────────
def hero_banner():
    st.markdown("""
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
          💻  AI-POWERED
      </p>
      <h1 style="font-family:'Syne',sans-serif; font-size:2.6rem; font-weight:800;
                 margin:0; color:#e8eaf6; line-height:1.15;">
          LaptopIQ
          <span style="
              background: linear-gradient(90deg,#6c63ff,#ff6584);
              -webkit-background-clip:text; -webkit-text-fill-color:transparent;
          ">Price Predictor</span>
      </h1>
      <p style="color:#8b91b8; margin:0.6rem 0 0; font-size:1.05rem;">
          Predict laptop prices intelligently using Machine Learning
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
 
 
def price_result_card(price):
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
                text-transform:uppercase; margin:0 0 0.5rem;">Estimated Price</p>
      <h1 style="
          font-family:'Syne',sans-serif; font-weight:800; font-size:3.2rem;
          background:linear-gradient(90deg,#43e97b,#6c63ff);
          -webkit-background-clip:text; -webkit-text-fill-color:transparent;
          margin:0;
      ">${price:,.0f}</h1>
      <p style="color:#8b91b8; margin:0.5rem 0 0; font-size:0.85rem;">
          ± estimated range: ${price*0.92:,.0f} – ${price*1.08:,.0f}
      </p>
    </div>
    """, unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.joblib")
        return model
    except FileNotFoundError:
        return None
 
 
# ─────────────────────────────────────────────
#  MOCK DATA  (replace with your real CSV)
# ─────────────────────────────────────────────
@st.cache_data
def load_sample_data():
    np.random.seed(42)
    n = 300
    brands   = ["Dell", "HP", "Lenovo", "Asus", "Acer", "Apple", "MSI", "Samsung"]
    procs    = ["Intel i3", "Intel i5", "Intel i7", "Intel i9", "AMD Ryzen 5", "AMD Ryzen 7", "Apple M1", "Apple M2"]
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
    st.markdown("""
    <div style="padding:1.2rem 0 0.5rem;">
      <p style="font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:800;
                color:#e8eaf6; margin:0;">💻 LaptopIQ</p>
      <p style="color:#8b91b8; font-size:0.8rem; margin:0;">ML Price Intelligence</p>
    </div>
    <hr style="border-color:#252840; margin:0.8rem 0;"/>
    """, unsafe_allow_html=True)
 
    page = st.radio(
        "Navigation",
        ["🏠  Home", "🔮  Predict Price", "📊  Analysis Dashboard", "⚖️  Compare Laptops"],
        label_visibility="collapsed",
    )
 
    st.markdown("""
    <hr style="border-color:#252840; margin:1rem 0;"/>
    <p style="color:#8b91b8; font-size:0.78rem;">
      Upload your <code style="color:#6c63ff;">model.joblib</code> in the same folder as this app.
    </p>
    """, unsafe_allow_html=True)
 
 
model = load_model()
df    = load_sample_data()
 
 
# ═══════════════════════════════════════════════════════════════════════
#  PAGE 1 – HOME
# ═══════════════════════════════════════════════════════════════════════
if "Home" in page:
    hero_banner()
 
    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_card("💻", "Laptops Analysed", f"{len(df):,}", "#6c63ff")
    with c2: stat_card("🏷️", "Avg Price",        f"${df.price.mean():,.0f}", "#ff6584")
    with c3: stat_card("📦", "Brands",            str(df.brand.nunique()), "#43e97b")
    with c4: stat_card("⭐", "Avg Rating",        f"{df.ratings.mean():.1f}", "#f7971e")
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    col_a, col_b = st.columns([1.4, 1])
 
    with col_a:
        section_title("Price Distribution", "How prices are spread across the dataset")
        fig = px.histogram(
            df, x="price", nbins=40,
            color_discrete_sequence=["#6c63ff"],
        )
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
        section_title("Top Brands", "Average price by brand")
        brand_avg = df.groupby("brand")["price"].mean().sort_values(ascending=True)
        fig2 = px.bar(
            brand_avg, orientation="h",
            color=brand_avg.values,
            color_continuous_scale=["#6c63ff", "#ff6584", "#f7971e"],
        )
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
elif "Predict" in page:
    hero_banner()
    section_title("🔮 Predict Laptop Price", "Fill in the specs and get an instant price estimate")
 
    if model is None:
        st.warning("⚠️  **model.joblib** not found – make sure it's in the same directory as this app. Showing a demo prediction instead.")
 
    col1, col2, col3 = st.columns(3)
 
    brands    = ["Dell", "HP", "Lenovo", "Asus", "Acer", "Apple", "MSI", "Samsung"]
    procs     = ["Intel i3", "Intel i5", "Intel i7", "Intel i9", "AMD Ryzen 5", "AMD Ryzen 7", "Apple M1", "Apple M2"]
    rams      = [4, 8, 16, 32, 64]
    storages  = [128, 256, 512, 1024, 2048]
 
    with col1:
        brand       = st.selectbox("Brand 🏷️",          brands)
        processor   = st.selectbox("Processor ⚡",       procs)
        ram         = st.selectbox("RAM (GB) 🧠",         rams, index=1)
        storage     = st.selectbox("Storage (GB) 💾",    storages, index=2)
 
    with col2:
        screen_size = st.slider("Screen Size (inches) 🖥️", 11.6, 17.3, 15.6, 0.1)
        cores       = st.selectbox("CPU Cores 🔲",          [2, 4, 6, 8, 12, 16], index=2)
        threads     = st.selectbox("CPU Threads 🔁",        [4, 8, 12, 16, 24, 32], index=2)
 
    with col3:
        ratings     = st.slider("User Rating ⭐",      1.0, 5.0, 4.2, 0.1)
        specs_score = st.slider("Specs Score 📈",       0,   100, 75)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    brand_enc = brands.index(brand)
    proc_enc  = procs.index(processor)
 
    input_data = np.array([[brand_enc, ratings, specs_score, proc_enc,
                            ram, storage, screen_size, 1920 * 1080,
                            cores, threads]])
 
    btn_col, _ = st.columns([1, 3])
    with btn_col:
        predict_btn = st.button("🚀  Predict Price", use_container_width=True)
 
    if predict_btn:
        if model:
            price = model.predict(input_data)[0]
        else:
            # demo formula when model is absent
            price = (ram * 15 + storage * 0.3 + specs_score * 12
                     + cores * 40 + ratings * 80 + 200 + brand_enc * 30)
 
        price_result_card(price)
 
        st.markdown("<br>", unsafe_allow_html=True)
        section_title("Similar Laptops in Dataset")
        similar = df[df["brand"] == brand].head(5)
        st.dataframe(
            similar[["brand", "processor", "ram", "storage", "screen_size", "ratings", "price"]],
            use_container_width=True, hide_index=True,
        )
 
 
# ═══════════════════════════════════════════════════════════════════════
#  PAGE 3 – DASHBOARD
# ═══════════════════════════════════════════════════════════════════════
elif "Analysis" in page:
    hero_banner()
    section_title("📊 Analysis Dashboard", "Insights from the laptop dataset")
 
    # Row 1 – Price vs RAM  |  Price vs Storage
    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(df, x="ram", y="price",
                     color="ram",
                     color_discrete_sequence=px.colors.sequential.Plasma)
        fig.update_layout(
            title="Price vs RAM", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="#8b91b8",
            xaxis=dict(gridcolor="#252840"), yaxis=dict(gridcolor="#252840"),
            showlegend=False, margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
 
    with c2:
        fig = px.box(df, x="storage", y="price",
                     color="storage",
                     color_discrete_sequence=px.colors.sequential.Viridis)
        fig.update_layout(
            title="Price vs Storage", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="#8b91b8",
            xaxis=dict(gridcolor="#252840"), yaxis=dict(gridcolor="#252840"),
            showlegend=False, margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
 
    # Row 2 – Scatter  |  Pie
    c3, c4 = st.columns(2)
    with c3:
        fig = px.scatter(df, x="specs_score", y="price", color="brand",
                         size="ram", hover_data=["processor"],
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(
            title="Specs Score vs Price (size = RAM)",
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
            title="Brand Distribution",
            paper_bgcolor="rgba(0,0,0,0)", font_color="#8b91b8",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
 
    # Row 3 – Heatmap correlation
    section_title("Correlation Matrix", "Relationship between numerical features")
    num_cols = ["ratings", "specs_score", "ram", "storage", "screen_size", "cores", "threads", "price"]
    corr = df[num_cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale=[[0,"#ff6584"], [0.5,"#252840"], [1,"#43e97b"]],
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
 
 
# ═══════════════════════════════════════════════════════════════════════
#  PAGE 4 – COMPARE
# ═══════════════════════════════════════════════════════════════════════
elif "Compare" in page:
    hero_banner()
    section_title("⚖️ Compare Laptops", "Select specs for two laptops and compare them side by side")
 
    brands    = ["Dell", "HP", "Lenovo", "Asus", "Acer", "Apple", "MSI", "Samsung"]
    procs     = ["Intel i3", "Intel i5", "Intel i7", "Intel i9", "AMD Ryzen 5", "AMD Ryzen 7", "Apple M1", "Apple M2"]
    rams      = [4, 8, 16, 32, 64]
    storages  = [128, 256, 512, 1024, 2048]
 
    colA, spacer, colB = st.columns([5, 0.5, 5])
 
    def laptop_form(col, label, color):
        with col:
            st.markdown(f"""
            <div style="background:#161929; border:1px solid {color}44;
                        border-radius:16px; padding:1.2rem 1.4rem; margin-bottom:1rem;">
              <p style="color:{color}; font-family:'Syne',sans-serif;
                        font-weight:800; font-size:1.1rem; margin:0;">{label}</p>
            </div>
            """, unsafe_allow_html=True)
            brand     = st.selectbox("Brand",     brands,   key=f"{label}_brand")
            proc      = st.selectbox("Processor", procs,    key=f"{label}_proc")
            ram       = st.selectbox("RAM (GB)",  rams,     key=f"{label}_ram",     index=2)
            storage   = st.selectbox("Storage",   storages, key=f"{label}_storage", index=2)
            cores     = st.selectbox("Cores",     [2,4,6,8,12,16], key=f"{label}_cores", index=2)
            ratings   = st.slider("Rating ⭐",  1.0, 5.0, 4.2, 0.1, key=f"{label}_rat")
            specs_sc  = st.slider("Specs Score", 0, 100, 75, key=f"{label}_sc")
            return {"brand": brand, "processor": proc, "ram": ram,
                    "storage": storage, "cores": cores,
                    "ratings": ratings, "specs_score": specs_sc}
 
    specs_A = laptop_form(colA, "💻 Laptop A", "#6c63ff")
    specs_B = laptop_form(colB, "💻 Laptop B", "#ff6584")
 
    compare_btn = st.button("⚖️  Compare Now", use_container_width=True)
 
    if compare_btn:
        def est_price(s):
            brand_enc = brands.index(s["brand"])
            proc_enc  = procs.index(s["processor"])
            inp = np.array([[brand_enc, s["ratings"], s["specs_score"], proc_enc,
                             s["ram"], s["storage"], 15.6, 1920*1080,
                             s["cores"], s["cores"]*2]])
            if model:
                return model.predict(inp)[0]
            return (s["ram"]*15 + s["storage"]*0.3 + s["specs_score"]*12
                    + s["cores"]*40 + s["ratings"]*80 + 200 + brand_enc*30)
 
        pA = est_price(specs_A)
        pB = est_price(specs_B)
 
        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1c1e35,#161929);
                        border:1px solid #6c63ff55; border-radius:16px;
                        padding:1.5rem; text-align:center;">
              <p style="color:#8b91b8;margin:0;font-size:0.85rem;">Laptop A – {specs_A['brand']}</p>
              <h2 style="font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;
                         background:linear-gradient(90deg,#6c63ff,#43e97b);
                         -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;">
                  ${pA:,.0f}
              </h2>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1c1e35,#161929);
                        border:1px solid #ff658455; border-radius:16px;
                        padding:1.5rem; text-align:center;">
              <p style="color:#8b91b8;margin:0;font-size:0.85rem;">Laptop B – {specs_B['brand']}</p>
              <h2 style="font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;
                         background:linear-gradient(90deg,#ff6584,#f7971e);
                         -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;">
                  ${pB:,.0f}
              </h2>
            </div>
            """, unsafe_allow_html=True)
 
        # Radar chart
        st.markdown("<br>", unsafe_allow_html=True)
        section_title("Radar Comparison")
 
        cats   = ["RAM", "Storage", "Cores", "Rating", "Specs Score"]
        maxv   = [64, 2048, 16, 5, 100]
        vals_A = [specs_A["ram"], specs_A["storage"], specs_A["cores"],
                  specs_A["ratings"], specs_A["specs_score"]]
        vals_B = [specs_B["ram"], specs_B["storage"], specs_B["cores"],
                  specs_B["ratings"], specs_B["specs_score"]]
 
        norm_A = [v/m*100 for v, m in zip(vals_A, maxv)]
        norm_B = [v/m*100 for v, m in zip(vals_B, maxv)]
 
        fig = go.Figure()
        for vals, name, color in [(norm_A, "Laptop A", "#6c63ff"), (norm_B, "Laptop B", "#ff6584")]:
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself", name=name,
                line_color=color, fillcolor=color.replace(")", ",0.15)").replace("rgb","rgba"),
            ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="#252840", color="#8b91b8"),
                angularaxis=dict(gridcolor="#252840", color="#8b91b8"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#8b91b8",
            showlegend=True,
            height=420,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
 
        diff = abs(pA - pB)
        winner = f"Laptop A ({specs_A['brand']})" if pA < pB else f"Laptop B ({specs_B['brand']})"
        st.info(f"💡 **{winner}** is cheaper by **${diff:,.0f}**")