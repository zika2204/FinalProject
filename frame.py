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

# Nâng cấp giao diện lên chuẩn "Hi-Tech Mystic" cực Wow
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@700;900&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    /* Nền ứng dụng chuyển màu mượt mà sâu thẳm */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background: radial-gradient(circle at top right, #FFFBF7 0%, #FAFAFA 100%);
    }

    /* Khối tiêu đề thương hiệu GIÀU HIỆU ỨNG WOW */
    .brand-container {
        text-align: center;
        padding: 40px 0 20px 0;
        animation: fadeInDown 1s ease-out;
    }
    .brand-title {
        font-family: 'Cinzel Decorative', serif;
        background: linear-gradient(135deg, #FF6F00 0%, #FFA000 50%, #E65100 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 4.5rem;
        margin-bottom: 0px;
        letter-spacing: 6px;
        /* Tạo hiệu ứng phát sáng mờ ảo cho chữ */
        filter: drop-shadow(0px 15px 20px rgba(230, 81, 0, 0.25));
        position: relative;
    }
    .brand-subtitle {
        color: #6E6E73;
        font-size: 1.15rem;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 20px;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    /* Nút bấm Camera và Dự đoán hiệu ứng Pulse 3D */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #FF6F00 0%, #E65100 100%);
        color: white;
        border-radius: 20px;
        border: none;
        font-weight: 700;
        padding: 1rem 2rem;
        width: 100%;
        font-size: 1.2rem;
        letter-spacing: 1px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0px 8px 30px rgba(230, 81, 0, 0.3);
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0px 15px 35px rgba(230, 81, 0, 0.5);
        color: white;
    }
    div.stButton > button:first-child:active {
        transform: translateY(-1px);
    }

    /* Hộp kết quả dạng Glassmorphism cao cấp */
    .prediction-box {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 111, 0, 0.08);
        border-left: 7px solid #FF6F00;
        padding: 28px;
        border-radius: 20px;
        margin-bottom: 25px;
        box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.02);
        transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
    }
    /* Hiệu ứng viền Neon cam phát sáng khi hover */
    .prediction-box:hover {
        transform: translateY(-5px);
        box-shadow: 0px 15px 40px rgba(255, 111, 0, 0.12);
        border-color: rgba(255, 111, 0, 0.3);
    }
    .prediction-title {
        color: #1D1D1F;
        font-weight: 700;
        font-size: 1.35rem;
        margin-bottom: 14px;
        letter-spacing: 0.5px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Thiết kế lại các Badge trạng thái sang xịn mịn */
    .status-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 4
