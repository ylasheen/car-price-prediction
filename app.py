import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# ─── Config ───
st.set_page_config(
    page_title="🚗 Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# ─── CSS ───
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .title {
        font-size: 48px;
        font-weight: bold;
        color: #00d4ff;
        text-align: center;
        padding: 20px;
    }
    .subtitle {
        font-size: 20px;
        color: #888;
        text-align: center;
        margin-bottom: 30px;
    }
    .price-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 2px solid #00d4ff;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    .price-value {
        font-size: 60px;
        font-weight: bold;
        color: #00d4ff;
    }
    .stButton>button {
        background: linear-gradient(135deg, #00d4ff, #0099cc);
        color: white;
        font-size: 20px;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 15px 40px;
        width: 100%;
        cursor: pointer;
    }
    .metric-card {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #333;
    }
    </style>
""", unsafe_allow_html=True)

# ─── تحميل الموديل ───
@st.cache_resource
def load_models():
    model = joblib.load('xgb_model.pkl')
    encoder = joblib.load('encoder.pkl')
    scaler = joblib.load('scaler.pkl')
    num_imputer = joblib.load('num_imputer.pkl')
    cat_imputer = joblib.load('cat_imputer.pkl')
    unique_values = joblib.load('unique_values.pkl')
    return model, encoder, scaler, num_imputer, cat_imputer, unique_values

model, encoder, scaler, num_imputer, cat_imputer, unique_values = load_models()

# ─── Header ───
st.markdown('<div class="title">🚗 Used Car Price Prediction</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">ادخل بيانات العربية وهنقولك سعرها بالذكاء الاصطناعي!</div>',
            unsafe_allow_html=True)

st.divider()

# ─── Sidebar ───
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774278.png",
             width=150)
    st.title("🚗 Car Price AI")
    st.write("---")
    st.info("""
    **عن الموديل:**
    - 🤖 XGBoost Regressor
    - 📊 R2 Score: 0.92
    - 🚗 تريننج على 300,000+ عربية
    - 📅 بيانات Craigslist الأمريكية
    """)
    st.write("---")
    st.success("done")

# ─── Inputs ───
st.subheader("📝 ادخل بيانات العربية")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔢 البيانات الرقمية**")
    year = st.slider('📅 سنة الصنع', 1990, 2024, 2015)
    odometer = st.number_input('🛣️ الكيلومترات', 0, 500000, 50000, step=1000)
    manufacturer = st.selectbox('🏭 الماركة',
                                  unique_values['manufacturer'])

with col2:
    st.markdown("**⚙️ المواصفات**")
    condition = st.selectbox('🔧 الحالة', unique_values['condition'])
    fuel = st.selectbox('⛽ نوع الوقود', unique_values['fuel'])
    transmission = st.selectbox('🔄 ناقل الحركة',
                                  unique_values['transmission'])

with col3:
    st.markdown("**📍 معلومات إضافية**")
    drive = st.selectbox('🚙 الدفع', unique_values['drive'])
    type_ = st.selectbox('🚗 النوع', unique_values['type'])
    state = st.selectbox('📍 الولاية', unique_values['state'])

st.divider()

# ─── Predict Button ───
if st.button('🔮 تنبأ بالسعر!'):

    # تجهيز البيانات
    input_df = pd.DataFrame([[year, odometer,
                               manufacturer, condition,
                               fuel, transmission,
                               drive, type_, state]],
                            columns=['year', 'odometer',
                                     'manufacturer', 'condition',
                                     'fuel', 'transmission',
                                     'drive', 'type', 'state'])

    num_cols = ['year', 'odometer']
    cat_cols = ['manufacturer', 'condition', 'fuel',
                'transmission', 'drive', 'type', 'state']

    input_df[num_cols] = num_imputer.transform(input_df[num_cols])
    input_df[cat_cols] = cat_imputer.transform(input_df[cat_cols])

    input_num = scaler.transform(input_df[num_cols])
    input_cat = encoder.transform(input_df[cat_cols])
    input_final = np.hstack([input_num, input_cat])

    prediction = model.predict(input_final)[0]

    # ─── النتيجة ───
    st.markdown(f"""
        <div class="price-box">
            <div style="color:#888; font-size:20px;">
                💰 السعر المتوقع للعربية
            </div>
            <div class="price-value">
                ${prediction:,.2f}
            </div>
            <div style="color:#888; font-size:16px;">
                دولار أمريكي
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ─── Metrics ───
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 السعر", f"${prediction:,.0f}")
    with col2:
        st.metric("📅 السنة", year)
    with col3:
        st.metric("🛣️ الكيلومترات", f"{odometer:,}")
    with col4:
        age = 2024 - year
        st.metric("⏰ عمر العربية", f"{age} سنة")

    st.divider()

    # ─── Charts ───
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 السعر مقارنة بالسنة")
        years = list(range(1990, 2025))
        prices = []
        for y in years:
            temp_df = pd.DataFrame([[y, odometer,
                                      manufacturer, condition,
                                      fuel, transmission,
                                      drive, type_, state]],
                                   columns=['year', 'odometer',
                                            'manufacturer', 'condition',
                                            'fuel', 'transmission',
                                            'drive', 'type', 'state'])
            temp_df[num_cols] = num_imputer.transform(temp_df[num_cols])
            temp_df[cat_cols] = cat_imputer.transform(temp_df[cat_cols])
            temp_num = scaler.transform(temp_df[num_cols])
            temp_cat = encoder.transform(temp_df[cat_cols])
            temp_final = np.hstack([temp_num, temp_cat])
            prices.append(model.predict(temp_final)[0])

        fig1 = px.line(
            x=years, y=prices,
            labels={'x': 'السنة', 'y': 'السعر ($)'},
            title='تأثير سنة الصنع على السعر'
        )
        fig1.add_vline(x=year, line_dash="dash",
                       line_color="red",
                       annotation_text=f"اختيارك: {year}")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("📊 السعر مقارنة بالكيلومترات")
        odometers = list(range(0, 300000, 10000))
        prices_od = []
        for od in odometers:
            temp_df = pd.DataFrame([[year, od,
                                      manufacturer, condition,
                                      fuel, transmission,
                                      drive, type_, state]],
                                   columns=['year', 'odometer',
                                            'manufacturer', 'condition',
                                            'fuel', 'transmission',
                                            'drive', 'type', 'state'])
            temp_df[num_cols] = num_imputer.transform(temp_df[num_cols])
            temp_df[cat_cols] = cat_imputer.transform(temp_df[cat_cols])
            temp_num = scaler.transform(temp_df[num_cols])
            temp_cat = encoder.transform(temp_df[cat_cols])
            temp_final = np.hstack([temp_num, temp_cat])
            prices_od.append(model.predict(temp_final)[0])

        fig2 = px.line(
            x=odometers, y=prices_od,
            labels={'x': 'الكيلومترات', 'y': 'السعر ($)'},
            title='تأثير الكيلومترات على السعر'
        )
        fig2.add_vline(x=odometer, line_dash="dash",
                       line_color="red",
                       annotation_text=f"اختيارك: {odometer:,}")
        st.plotly_chart(fig2, use_container_width=True)

    # ─── Feature Importance ───
    st.subheader("🎯 أهم العوامل في تحديد السعر")
    
    feature_names = (num_cols + 
                     list(encoder.get_feature_names_out(cat_cols)))
    importances = model.feature_importances_
    
    top_idx = np.argsort(importances)[-15:]
    top_features = [feature_names[i] for i in top_idx]
    top_importances = [importances[i] for i in top_idx]

    fig3 = px.bar(
        x=top_importances,
        y=top_features,
        orientation='h',
        title='Top 15 Feature Importance',
        labels={'x': 'Importance', 'y': 'Feature'},
        color=top_importances,
        color_continuous_scale='blues'
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ─── Footer ───
    st.divider()
    st.markdown("""
        <div style='text-align: center; color: #888;'>
            🤖 Powered by XGBoost & Streamlit | 
            📊 Trained on 300,000+ Cars
        </div>
    """, unsafe_allow_html=True)