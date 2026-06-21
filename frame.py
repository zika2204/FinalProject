import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

st.set_page_config(
    page_title="PalmID - Xem Bói Chỉ Tay AI",
    page_icon="🔮",
    layout="centered"
)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #FAFAFA;
    }
    .brand-container {
        text-align: center;
        padding: 20px 0 10px 0;
    }
    .brand-title {
        font-family: 'Cinzel', serif;
        background: linear-gradient(135deg, #FF6F00 0%, #FFA000 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.8rem;
        margin-bottom: 0px;
        letter-spacing: 3px;
        filter: drop-shadow(0px 4px 8px rgba(230, 81, 0, 0.2));
    }
    .brand-subtitle {
        color: #555555;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 35px;
        letter-spacing: 0.5px;
    }

    /* Nút bấm Camera và Dự đoán hiệu ứng mềm mại */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #FF6F00 0%, #E65100 100%);
        color: white;
        border-radius: 16px;
        border: none;
        font-weight: 700;
        padding: 0.85rem 2rem;
        width: 100%;
        font-size: 1.15rem;
        transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
        box-shadow: 0px 6px 20px rgba(230, 81, 0, 0.25);
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-3px);
        box-shadow: 0px 8px 25px rgba(230, 81, 0, 0.45);
        color: white;
    }
    div.stButton > button:first-child:active {
        transform: translateY(-1px);
    }

    .prediction-box {
        background: #FFFFFF;
        border-left: 6px solid #FF6F00;
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 22px;
        box-shadow: 0px 4px 18px rgba(0, 0, 0, 0.04);
        transition: all 0.3s ease;
    }
    .prediction-box:hover {
        transform: scale(1.01);
        box-shadow: 0px 8px 25px rgba(255, 111, 0, 0.08);
    }
    .prediction-title {
        color: #E65100;
        font-weight: 700;
        font-size: 1.25rem;
        margin-bottom: 12px;
        letter-spacing: 0.5px;
    }
    
    .status-badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-short { background-color: #FFF3E0; color: #E65100; border: 1px solid #FFE0B2; }
    .badge-medium { background-color: #FFF3E0; color: #FB8C00; border: 1px solid #FFE0B2; }
    .badge-long { background-color: #E65100; color: white; }
    .prediction-text {
        font-size: 1rem;
        line-height: 1.6;
        color: #333333;
        margin-top: 8px;
    }
    .advice-text {
        font-size: 1rem;
        line-height: 1.6;
        color: #D35400;
        font-weight: 600;
        background-color: #FFF8F2;
        padding: 12px 16px;
        border-radius: 10px;
        margin-top: 12px;
        border-left: 3px solid #FF9800;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class='brand-container'>
        <div class='brand-title'>PalmID</div>
        <div class='brand-subtitle'>🔮 Hệ Thống Trích Xuất Vận Mệnh & Xem Bói Chỉ Tay Bằng AI 🔮</div>
    </div>
""", unsafe_allow_html=True)

MODEL_PATH = 'ai.h5'
IMG_SIZE = (128, 128)

@st.cache_resource
def load_my_model():
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    else:
        st.error(f"Không tìm thấy file mô hình '{MODEL_PATH}'. Vui lòng kiểm tra lại trên kho lưu trữ GitHub của bạn.")
        return None

model = load_my_model()

def generate_fortune(sunghiep, tridao, tamdao, sinhdao):
    fortunes = {}
    if sunghiep < 0.35:
        fortunes['sunghiep'] = {
            "status": "Đang định hình (Ngắn/Mờ)",
            "class": "badge-short",
            "meaning": "Đường sự nghiệp mờ hoặc ngắn phản ánh bạn đang ở điểm khởi đầu hoặc đang trăn trở tìm kiếm định hướng thực sự của cuộc đời. Bạn có nhiều ý tưởng nhưng chưa thể hội tụ thành một lối đi cố định lâu dài.",
            "advice": "Giai đoạn này đòi hỏi sự kiên nhẫn tích lũy. Đừng vội vã nhảy việc chỉ vì cảm thấy chán nản lâm thời. Hãy tập trung mài giũa một kỹ năng chuyên môn sâu mà bạn giỏi nhất để làm bước đệm vững chắc."
        }
    elif sunghiep < 0.70:
        fortunes['sunghiep'] = {
            "status": "Ổn định, vững vàng (Trung bình)",
            "class": "badge-medium",
            "meaning": "Bạn sở hữu đường sự nghiệp có độ dài vừa vặn, thể hiện một lộ trình công danh tiến triển tuần tự, rõ ràng và ít gặp biến cố lớn. Bạn là người làm việc có kế hoạch, nhận được sự tín nhiệm từ cấp trên.",
            "advice": "An toàn quá mức đôi khi sẽ làm giảm sức bật. Lời khuyên dành cho bạn là hãy can đảm bước ra ngoài vùng an toàn hiện tại, chủ động đón nhận các dự án khó hoặc mang tính đổi mới để bứt phá lên vị trí cao hơn."
        }
    else:
        fortunes['sunghiep'] = {
            "status": "Rực rỡ, thủ lĩnh (Dài/Rõ nét)",
            "class": "badge-long",
            "meaning": "Đường sự nghiệp dài, đậm nét chứng tỏ bạn mang số mệnh của người làm chủ, quyết đoán và có ý chí kiên cường bẩm sinh. Bạn dễ khẳng định được chỗ đứng vững chắc và đạt thành tựu tài chính lớn trước tuổi 35.",
            "advice": "Tướng làm chủ thường đi kèm với áp lực tinh thần rất cao. Bạn cần chú ý lắng nghe ý kiến đóng góp từ những người xung quanh và học cách phân chia thời gian hợp lý để tránh rơi vào trạng thái kiệt sức."
        }
    if tridao < 0.40:
        fortunes['tridao'] = {
            "status": "Thực tế, nhạy bén (Ngắn/Mờ)",
            "class": "badge-short",
            "meaning": "Chỉ số trí đạo ở mức thấp cho thấy bạn là người có tư duy trực diện, xử lý vấn đề dựa vào trực giác và tính ứng dụng thực tế cao. Bạn ghét những lý thuyết suông và muốn bắt tay vào hành động ngay.",
            "advice": "Hành động nhanh là lợi thế lớn, tuy nhiên với các quyết định hệ trọng liên quan tới tài chính hoặc hợp đồng dài hạn, bạn bắt buộc phải kiềm chế sự nóng nội, dành thêm thời gian phân tích rủi ro đa chiều."
        }
    elif tridao < 0.70:
        fortunes['tridao'] = {
            "status": "Cân bằng, linh hoạt (Trung bình)",
            "class": "badge-medium",
            "meaning": "Đường trí đạo có độ dài lý tưởng, biểu thị một trí tuệ linh hoạt, dung hòa tốt giữa lý thuyết khoa học và thực tiễn đời sống. Bạn có năng lực tiếp thu kiến thức mới một cách bài bản và giữ được bình tĩnh trước áp lực.",
            "advice": "Hãy duy trì thói quen đọc sách và học hỏi các kỹ năng bổ trợ (như ngoại ngữ, quản lý tài chính). Đây là chìa khóa giúp năng lực tư duy tổng hợp của bạn luôn dẫn đầu xu hướng thị trường."
        }
    else:
        fortunes['tridao'] = {
            "status": "Sâu sắc, bác học (Dài/Rõ nét)",
            "class": "badge-long",
            "meaning": "Bạn có một tư duy logic cực kỳ xuất sắc, khả năng tập trung cao độ và có xu hướng đào sâu nghiên cứu bản chất của mọi sự vật. Tuy nhiên, bạn cũng là người dễ bị rơi vào trạng thái suy nghĩ quá nhiều (overthinking).",
            "advice": "Suy nghĩ sâu sắc là món quà, nhưng đừng biến nó thành rào cản hành động. Hãy rèn luyện kỹ năng thả lỏng tâm trí, thực hành thiền định hoặc các bộ môn thể thao giải tỏa căng thẳng để đầu óc không bị quá tải."
        }
    if tamdao < 0.40:
        fortunes['tamdao'] = {
            "status": "Lý trí, kín kẽ (Ngắn/Mờ)",
            "class": "badge-short",
            "meaning": "Đường tâm đạo mờ thể hiện xu hướng đặt lý trí lên trên cảm xúc trong mọi mối quan hệ. Bạn kiểm soát nội tâm rất chặt chẽ, yêu ghét phân minh, không thích thể hiện tình cảm quá đà ra bên ngoài.",
            "advice": "Việc giữ sự lạnh lùng cần thiết giúp bạn ít bị lợi dụng, song nó có thể tạo khoảng cách vô hình với người thân thiết. Hãy học cách mở lòng, lắng nghe và chia sẻ những vụn vặt cảm xúc để mối quan hệ được thăng hoa."
        }
    elif tamdao < 0.70:
        fortunes['tamdao'] = {
            "status": "Hài hòa, chân thành (Trung bình)",
            "class": "badge-medium",
            "meaning": "Chỉ số tâm đạo vừa vặn biểu thị một đời sống tinh thần lành mạnh, biết yêu thương trọn vẹn nhưng không mù quáng. Bạn là người bạn đời, người đồng hành đáng tin cậy nhờ sự thấu hiểu và biết nhường nhịn.",
            "advice": "Để duy trì ngọn lửa hạnh phúc lâu dài, hãy luôn duy trì thói quen giao tiếp thẳng thắn và cùng đối phương xây dựng những mục tiêu chung rõ ràng thay vì chỉ giữ sự im lặng nhẫn nhịn."
        }
    else:
        fortunes['tamdao'] = {
            "status": "Giàu cảm xúc, sâu sắc (Dài/Rõ nét)",
            "class": "badge-long",
            "meaning": "Mô hình nhận diện đường tâm đạo của bạn rất phát triển, cho thấy bạn là người sống nồng nhiệt, nhạy cảm và luôn sẵn sàng hy sinh hết mình cho người mình yêu. Bạn dễ bị ảnh hưởng tâm lý mạnh mẽ bởi vui buồn của người khác.",
            "advice": "Khi bạn trao đi quá nhiều, bạn sẽ dễ bị tổn thương sâu sắc khi thực tế không như kỳ vọng. Lời khuyên lớn nhất là hãy thiết lập ranh giới cảm xúc và học cách yêu thương, trân trọng bản thân trước khi kỳ vọng điều đó từ người khác."
        }
    if sinhdao < 0.50:
        fortunes['sinhdao'] = {
            "status": "Cơ địa nhạy cảm (Ngắn/Mờ)",
            "class": "badge-short",
            "meaning": "Đường sinh đạo mờ phản ánh năng lượng thể chất của bạn dễ bị hao hụt, sức bền không cao và dễ chịu tác động bởi thời tiết thất thường hoặc áp lực từ lịch trình làm việc căng thẳng.",
            "advice": "Đừng lo lắng về tuổi thọ vì độ dài sinh đạo chỉ biểu thị năng lượng sống lực. Bạn cần khẩn cấp thiết lập lại đồng hồ sinh học: ăn uống đủ chất, ngủ đủ giấc và duy trì một bài tập vận động nhẹ nhàng (như yoga, đi bộ) hàng ngày."
        }
    else:
        fortunes['sinhdao'] = {
            "status": "Sinh lực dồi dào (Dài/Rõ nét)",
            "class": "badge-long",
            "meaning": "Mô hình AI nhận diện đường sinh đạo của bạn rất sâu rộng, thể hiện một nền tảng thể chất tuyệt vời, khả năng phục hồi tổn thương cực nhanh cùng một tinh thần sống lạc quan, tràn đầy nhiệt huyết.",
            "advice": "Sở hữu cơ địa tốt là một lợi thế vàng, nhưng tuyệt đối không được chủ quan. Hãy loại bỏ các thói quen xấu như thức khuya triền miên hay lạm dụng các chất kích thích để bảo vệ nguồn năng lượng quý giá này lâu dài."
        }
        
    return fortunes

if model is not None:
    img_file_buffer = st.camera_input("Vui lòng giữ bàn tay thẳng, rõ nét trước ống kính camera")

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        img_resized = image.resize(IMG_SIZE)
        img_array = np.array(img_resized) / 255.0
        img_input = np.expand_dims(img_array, axis=0)
        
        if st.button("🔮 BẮT ĐẦU PHÂN TÍCH VẬN MỆNH 🔮"):
            with st.spinner("Hệ thống PalmID đang quét cấu trúc đường chỉ tay..."):
                predictions = model.predict(img_input)[0]
                sunghiep, tridao, tamdao, sinhdao = np.clip(predictions, 0.0, 1.0)
                result_fortune = generate_fortune(sunghiep, tridao, tamdao, sinhdao)
                
                st.success("✨ Đã phân tích bản đồ bàn tay thành công! Dưới đây là kết quả luận giải chi tiết từ PalmID:")
                st.write("---")
                keys = ['sunghiep', 'tridao', 'tamdao', 'sinhdao']
                display_names = ["💼 ĐƯỜNG SỰ NGHIỆP", "🧠 ĐƯỜNG TRÍ ĐẠO", "❤️ ĐƯỜNG TÂM ĐẠO", "🌱 ĐƯỜNG SINH ĐẠO"]
                
                for key, name in zip(keys, display_names):
                    data = result_fortune[key]
                    score = predictions[keys.index(key)]
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div class="prediction-title">{name} <span style='color: #888; font-size: 0.9rem; font-weight: normal;'>(Chỉ số nét: {score:.2f})</span></div>
                        <span class="status-badge {data['class']}">{data['status']}</span>
                        <div class="prediction-text"><b>Luận giải:</b> {data['meaning']}</div>
                        <div class="advice-text"><b>💡 Lời khuyên định hướng:</b> {data['advice']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.balloons()
