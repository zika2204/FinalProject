import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

st.set_page_config(
    page_title="Xem Bói Chỉ Tay AI",
    page_icon="✋",
    layout="centered"
)

st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #FF6F00;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        padding: 0.5rem 2rem;
        width: 100%;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #E65100;
        color: white;
        box-shadow: 0px 4px 10px rgba(255, 111, 0, 0.4);
    }
    .main-title {
        color: #E65100;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .prediction-box {
        background-color: #FFF3E0;
        border-left: 5px solid #FF6F00;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .prediction-title {
        color: #E65100;
        font-weight: bold;
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🔮 HỆ THỐNG XEM BÓI CHỈ TAY AI 🔮</h1>", unsafe_allow_html=True)
st.write("Đưa bàn tay của bạn vào khung camera bên dưới, hệ thống AI sẽ phân tích các đường nét để luận giải vận mệnh.")

MODEL_PATH = 'chitay.h5'
IMG_SIZE = (128, 128)

@st.cache_resource
def load_my_model():
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    else:
        st.error(f"Không tìm thấy file mô hình '{MODEL_PATH}'. Vui lòng để file model cùng thư mục với app.py")
        return None

model = load_my_model()

def generate_fortune(sunghiep, tridao, tamdao, sinhdao):
    fortunes = {}
    
    # 1. Sự nghiệp
    if sunghiep < 0.3:
        fortunes['sunghiep'] = {
            "status": "Đang định hình",
            "meaning": "Đường sự nghiệp mờ hoặc ngắn cho thấy bạn đang trong giai đoạn tìm kiếm hướng đi, có nhiều ngã rẽ và chưa thực sự ổn định vào một công việc cố định.",
            "advice": "Đừng quá nôn nóng đổi việc. Hãy tập trung tích lũy kiến thức nền tảng và tìm ra thế mạnh cốt lõi của bản thân trước."
        }
    elif sunghiep < 0.7:
        fortunes['sunghiep'] = {
            "status": "Vững vàng, ổn định",
            "meaning": "Bạn có một lộ trình sự nghiệp rõ ràng, công việc tiến triển đều đặn. Bạn là người chăm chỉ và được đồng nghiệp tin tưởng.",
            "advice": "Học cách bước ra khỏi vùng an toàn. Hãy chủ động nhận các dự án khó hơn để có cơ hội thăng tiến đột phá."
        }
    else:
        fortunes['sunghiep'] = {
            "status": "Rực rỡ, có tướng làm chủ",
            "meaning": "Đường sự nghiệp rất dài và rõ nét. Bạn sở hữu tố chất lãnh đạo bẩm sinh, ý chí kiên cường và dễ đạt thành tựu lớn trước tuổi 35.",
            "advice": "Đi kèm với quyền lực là áp lực. Hãy chú ý giữ mối quan hệ tốt với cấp dưới và cân bằng giữa công việc và gia đình."
        }

    # 2. Trí đạo (Trí tuệ)
    if tridao < 0.4:
        fortunes['tridao'] = {
            "status": "Thực tế, hành động nhanh",
            "meaning": "Bạn là người thiên về hành động, thích sự thực tế và có tư duy trực diện. Bạn không thích những lý thuyết suông.",
            "advice": "Trước khi đưa ra các quyết định lớn liên quan đến tiền bạc, hãy dành thêm thời gian suy nghĩ kỹ để tránh sai sót."
        }
    else:
        fortunes['tridao'] = {
            "status": "Sâu sắc, tư duy logic cao",
            "meaning": "Đường trí đạo dài chứng tỏ bạn có khả năng tập trung cao, tư duy phân tích tốt và rất nhạy bén với các thông tin mới.",
            "advice": "Đôi khi bạn hay bị suy nghĩ quá nhiều (overthinking). Hãy học cách thả lỏng tâm trí và thiền định."
        }

    # 3. Tâm đạo (Tình duyên)
    if tamdao < 0.4:
        fortunes['tamdao'] = {
            "status": "Kín đáo, lý trí trong tình cảm",
            "meaning": "Bạn kiểm soát cảm xúc rất tốt, yêu ghét rõ ràng và không để chuyện tình cảm làm ảnh hưởng đến lý trí của mình.",
            "advice": "Hãy mở lòng hơn và chia sẻ tâm tư với đối phương, sự lạnh lùng đôi khi tạo ra khoảng cách vô hình."
        }
    else:
        fortunes['tamdao'] = {
            "status": "Giàu cảm xúc, chân thành",
            "meaning": "Đường tâm đạo sâu rộng thể hiện bạn là người sống tình cảm, biết quan tâm và luôn hết mình vì người khác khi yêu.",
            "advice": "Sống quá tình cảm dễ khiến bạn bị tổn thương bởi người khác. Hãy học cách yêu thương bản thân mình trước tiên."
        }

    # 4. Sinh đạo (Sức khỏe)
    if sinhdao < 0.5:
        fortunes['sinhdao'] = {
            "status": "Năng lượng dễ hao hụt",
            "meaning": "Đường sinh đạo ngắn không có nghĩa là tuổi thọ ngắn (đây là quan niệm sai lầm), nó chỉ ra rằng sức bền và thể trạng của bạn dễ bị ảnh hưởng bởi thời tiết hoặc stress.",
            "advice": "Hãy thiết lập lại chế độ ăn uống, ngủ nghỉ đúng giờ và rèn luyện thể thao đều đặn để nâng cao sức đề kháng."
        }
    else:
        fortunes['sinhdao'] = {
            "status": "Sức sống dồi dào, dẻo dai",
            "meaning": "Bạn sở hữu một nguồn năng lượng thể chất tuyệt vời, khả năng phục hồi sau đau ốm rất nhanh và tinh thần luôn tích cực.",
            "advice": "Dù khỏe mạnh nhưng đừng chủ quan làm việc quá sức hay lạm dụng các chất kích thích, thức khuya."
        }
        
    return fortunes
if model is not None:
    img_file_buffer = st.camera_input("Đặt bàn tay của bạn thẳng trước camera")

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        img_resized = image.resize(IMG_SIZE)
        img_array = np.array(img_resized) / 255.0
        img_input = np.expand_dims(img_array, axis=0) # Thêm chiều batch (1, 128, 128, 3)
        if st.button("🔮 BẮT ĐẦU DỰ ĐOÁN VẬN MỆNH 🔮"):
            with st.spinner("AI đang quét các nét chỉ tay..."):
                predictions = model.predict(img_input)[0]
                sunghiep, tridao, tamdao, sinhdao = np.clip(predictions, 0.0, 1.0)
                result_fortune = generate_fortune(sunghiep, tridao, tamdao, sinhdao)
                
                st.success("Phân tích thành công! Dưới đây là kết quả của bạn:")
                st.write("---")
                
                keys = ['sunghiep', 'tridao', 'tamdao', 'sinhdao']
                display_names = ["💼 ĐƯỜNG SỰ NGHIỆP", "🧠 ĐƯỜNG TRÍ ĐẠO", "❤️ ĐƯỜNG TÂM ĐẠO", "🌱 ĐƯỜNG SINH ĐẠO"]
                
                for key, name in zip(keys, display_names):
                    data = result_fortune[key]
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div class="prediction-title">{name} (Độ rõ nét: {predictions[keys.index(key)]:.2f})</div>
                        <p><b>Trạng thái:</b> {data['status']}</p>
                        <p><b>Luận giải:</b> {data['meaning']}</p>
                        <p style='color: #E65100;'><b>💡 Lời khuyên:</b> {data['advice']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.balloons() 
