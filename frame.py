import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# ==========================================
# 0. HÀM TRẢ VỀ CSS PREMIUM (ĐẶC BIỆT CHỐNG LỖI CHUỖI)
# ==========================================
def get_css():
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@700;900&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background: radial-gradient(circle at top right, #FFFBF7 0%, #FAFAFA 100%);
    }

    .brand-container {
        text-align: center;
        padding: 30px 0 10px 0;
    }

    .brand-title {
        font-family: 'Cinzel', serif;
        background: linear-gradient(135deg, #FF6F00 0%, #FFA000 50%, #E65100 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 4.2rem;
        margin-bottom: 0px;
        letter-spacing: 4px;
        filter: drop-shadow(0px 10px 15px rgba(230, 81, 0, 0.2));
    }

    .brand-subtitle {
        color: #6E6E73;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 20px;
        letter-spacing: 1px;
    }

    div.stButton > button:first-child {
        background: linear-gradient(135deg, #FF6F00 0%, #E65100 100%);
        color: white !important;
        border-radius: 16px;
        border: none;
        font-weight: 700;
        padding: 0.9rem 2rem;
        width: 100%;
        font-size: 1.15rem;
        transition: all 0.3s ease;
        box-shadow: 0px 6px 25px rgba(230, 81, 0, 0.25);
    }

    div.stButton > button:first-child:hover {
        transform: translateY(-3px);
        box-shadow: 0px 12px 30px rgba(230, 81, 0, 0.45);
        color: white !important;
    }

    .prediction-box {
        background: #FFFFFF;
        border-left: 7px solid #FF6F00;
        padding: 24px;
        border-radius: 18px;
        margin-bottom: 20px;
        box-shadow: 0px 8px 25px rgba(0, 0, 0, 0.03);
        transition: all 0.3s ease;
    }

    .prediction-box:hover {
        transform: translateY(-3px);
        box-shadow: 0px 12px 30px rgba(255, 111, 0, 0.1);
    }

    .prediction-title {
        color: #1D1D1F;
        font-weight: 700;
        font-size: 1.3rem;
        margin-bottom: 12px;
    }

    .status-badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 40px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 15px;
        text-transform: uppercase;
    }

    .badge-short { background: #FFF3E0; color: #E65100; }
    .badge-medium { background: #FFF8F2; color: #FB8C00; }
    .badge-long { background: linear-gradient(135deg, #FF6F00 0%, #E65100 100%); color: white; }

    .prediction-text {
        font-size: 1rem;
        line-height: 1.6;
        color: #3A3A3C;
        margin-top: 8px;
    }

    .advice-text {
        font-size: 1rem;
        line-height: 1.6;
        color: #D35400;
        font-weight: 600;
        background: #FFF9F5;
        padding: 12px 18px;
        border-radius: 12px;
        margin-top: 14px;
        border-left: 4px solid #FF6F00;
    }
    </style>
    """

# ==========================================
# 1. CẤU HÌNH CONFIG & INJECT STYLE
# ==========================================
st.set_page_config(
    page_title="PalmID - Khám Phá Vận Mệnh AI",
    page_icon="🔮",
    layout="centered"
)

# Thực thi CSS an toàn
st.markdown(get_css(), unsafe_allow_html=True)

# Khối thương hiệu hoàng gia (Đã chuyển sang chuỗi một dòng an toàn tuyệt đối)
st.markdown("<div class='brand-container'><div class='brand-title'>PalmID</div><div class='brand-subtitle'>✨ HỆ THỐNG TRÍCH XUẤT VẬN MỆNH THƯỢNG LƯU BẰNG AI ✨</div></div>", unsafe_allow_html=True)

# Kích khởi các trạng thái bộ nhớ ban đầu nếu chưa có
if 'analyzed' not in st.session_state:
    st.session_state['analyzed'] = False

# ==========================================
# 2. TẢI MÔ HÌNH H5
# ==========================================
MODEL_PATH = 'chitay_compact.h5'
IMG_SIZE = (128, 128)

@st.cache_resource
def load_my_model():
