import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN PREMIUM LUXURY (PALMID)
# ==========================================
st.set_page_config(
    page_title="PalmID - Khám Phá Vận Mệnh AI",
    page_icon="🔮",
    layout="centered"
)

# Fix triệt để các ký tự đặc biệt trong CSS giúp Streamlit không bị lỗi biên dịch
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@700;900&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    /* Nền ứng dụng chuyển màu mượt mà sâu thẳm */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background: radial-gradient(circle at top right, #FFFBF7 0%, #FAFAFA 100%);
    }

    /* Khối tiêu đề thương hiệu GIÀU HIỆU ỨNG WOW */
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

    /* Nút bấm Camera và Dự đoán hiệu ứng Pulse 3D */
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

    /* Hộp kết quả dạng Glassmorphism cao cấp */
    .prediction-box {
        background: #FFFFFF;
        border-left: 7px solid #FF6F00;
        padding: 24px;
        border-radius: 18px;
        margin-bottom: 20px;
        box-shadow: 0px 8px 25px rgba(0, 0, 0, 0.03);
        transition: all 0.3s ease;
    }
    /* Hiệu ứng viền phát sáng nhẹ khi hover */
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
    
    /* Thiết kế lại các Badge trạng thái sang xịn mịn */
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
    .badge-long { background: linear-gradient(135deg
