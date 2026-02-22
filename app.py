import streamlit as st
import joblib
import numpy as np
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="HDFC Loan AI Assistant", page_icon="🏦", layout="wide")

# --- 2. PROFESSIONAL HDFC THEME CSS ---
st.markdown("""
    <style>
    /* Main Background and Text Contrast */
    .stApp {
        background-color: #f4f7f9;
        color: #1e1e1e;
    }
    
    /* Headers Styling */
    h1, h2, h3 {
        color: #004684 !important;
        font-weight: 700 !important;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #004684 !important;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Cards and Metric Boxes */
    .metric-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 6px solid #ed1c24;
        margin-bottom: 20px;
    }

    /* Logic Box on Home Page */
    .logic-box {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 15px;
        border-left: 10px solid #004684;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
        color: #333333 !important;
    }

    /* HDFC Red Button */
    div.stButton > button {
        background-color: #ed1c24 !important;
        color: #ffffff !important;
        font-size: 18px !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        padding: 12px !important;
        width: 100% !important;
        border: none !important;
    }
    div.stButton > button:hover {
        background-color: #004684 !important;
        border: 1px solid white !important;
    }

    /* Team Page: Lead vs Member Cards */
    .lead-card {
        background-color: #ffffff;
        padding: 40px !important;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        text-align: center;
        border: 2px solid #ed1c24;
        border-top: 10px solid #004684;
    }
    .member-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        text-align: center;
        border-top: 5px solid #004684;
    }

    /* Labels Contrast */
    label {
        color: #333333 !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# --- 4. NAVIGATION ---
page = st.sidebar.radio("Main Menu", ["🏠 Home: The Logic", "📊 Loan Predictor", "👥 Meet the Team"])

# ==========================================
# PAGE 1: HOME (MAGIC OF LOGISTIC REGRESSION)
# ==========================================
if page == "🏠 Home: The Logic":
    st.markdown("<h1 style='text-align: center;'>🏦 HDFC Loan AI Approval System</h1>", unsafe_allow_html=True)
    st.write("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-card'><h1>98.7%</h1><p>Model Accuracy</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><h2>Random Forest</h2><p>Core Algorithm</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'><h2>9 Optimized</h2><p>Data Features</p></div>", unsafe_allow_html=True)

    st.markdown("### 🧠 Why Random Forest was the Best Choice?")
    cl, cr = st.columns([1.5, 1])
    with cl:
        st.markdown("""
        <div class='logic-box'>
        <b>1. Binary :</b> It is considered the gold standard for loan decisions (Yes/No) because it provides classification probabilities..<br><br>
        <b>2. Mathematically Interpretable:</b> For HDFC’s risk assessment, it is important to understand how <b>CIBIL</b> and <b>Total Assets</b> influenced the decision-making process.<br><br>
        <b>3. Scalable Logic:</b> We simplified the total wealth by calculating the sum of assets, which Random Forest can process very efficiently.
        </div>
        """, unsafe_allow_html=True)
    with cr:
        st.image("https://img.freepik.com/free-vector/bank-manager-concept-illustration_114360-7023.jpg")

# ==========================================
# PAGE 2: LOAN PREDICTOR (9-FEATURE LOGIC)
# ==========================================
elif page == "📊 Loan Predictor":
    st.markdown("<h2>📊 AI Eligibility Predictor</h2>", unsafe_allow_html=True)
    
    if model is None or scaler is None:
        st.error("❌ 'model.pkl' ya 'scaler.pkl' missing hai!")
        st.stop()

    with st.container():
        with st.form("loan_form"):
            c1, c2 = st.columns(2)
            with c1:
                dependents = st.number_input("Dependents", 0, 10, 1)
                edu = st.selectbox("Education", ["Graduate", "Not Graduate"])
                emp = st.selectbox("Self Employed?", ["Yes", "No"])
                income = st.number_input("Annual Income (₹)", value=600000)
                loan_amt = st.number_input("Loan Amount (₹)", value=300000)
            
            with c2:
                term = st.number_input("Term (Years)", 1, 30, 5)
                cibil = st.slider("CIBIL Score", 300, 900, 750)
                bank_assets = st.number_input("Bank Asset Value (₹)", value=100000)
                st.write("---")
                res = st.number_input("Residential Asset Value", value=500000)
                comm = st.number_input("Commercial Asset Value", value=0)
                lux = st.number_input("Luxury Asset Value", value=100000)
            
            submit = st.form_submit_button("Check Approval Status")

    if submit:
        # Asset Summation Logic
        total_assets = res + comm + lux
        edu_val = 1 if edu == "Graduate" else 0
        emp_val = 1 if emp == "Yes" else 0
        
        # Exact Column Order for Model [9 Features]:
        # [dep, edu, emp, inc, l_amt, term, cibil, bank, total]
        raw_input = np.array([[dependents, edu_val, emp_val, income, loan_amt, term, cibil, bank_assets, total_assets]])
        
        with st.spinner("AI Analysis in progress..."):
            time.sleep(1)
            # Scaling data before prediction
            scaled_input = scaler.transform(raw_input)
            prediction = model.predict(scaled_input)
            
            st.markdown("---")
            if prediction[0] == 1:
                st.balloons()
                st.markdown("<div style='background:#d4edda; padding:20px; border-radius:10px; color:#155724; text-align:center; font-size:24px; font-weight:bold;'>✅ LOAN APPROVED</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='background:#f8d7da; padding:20px; border-radius:10px; color:#721c24; text-align:center; font-size:24px; font-weight:bold;'>❌ LOAN REJECTED</div>", unsafe_allow_html=True)

# ==========================================
# PAGE 3: TEAM (PROMINENT LEAD BOX)
# ==========================================
elif page == "👥 Meet the Team":
    st.markdown("<h1 style='text-align: center;'>👥 The Project Visionaries</h1>", unsafe_allow_html=True)
    st.write("---")
    
    # Akshat Gupta (Team Lead) ka box bada karne ke liye ratio 1.5 kiya hai
    col_lead, col2, col3, col4 = st.columns([1.5, 1, 1, 1])
    
    with col_lead:
        st.markdown(f"""
            <div class='lead-card'>
                <h2 style='margin-bottom:0;'>Akshat Gupta</h2>
                <h4 style='color: #ed1c24; margin-top:0;'>🚀 Team Lead</h4>
                <p>Lead ML Engineer & Architect</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='member-card'><h3>Mohak Sharma</h3><p>Data Scientist</p></div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='member-card'><h3>Jasmine Kaur</h3><p>ML Engineer</p></div>", unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='member-card'><h3>Manav Kaushik</h3><p>Risk Analyst</p></div>", unsafe_allow_html=True)


    st.write("<br><br><p style='text-align: center; color: gray;'>Computer Engineering Project | Thapar University</p>", unsafe_allow_html=True)


