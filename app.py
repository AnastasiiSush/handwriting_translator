import streamlit as st
import cv2
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from streamlit import camera_input
import json
import easyocr

from data_utils import processing_image, detect_crop_text
from predict import ctc_best_path_decoding, get_final_word

st.set_page_config(page_title="Handwritten Text Recognizer", page_icon="📝")
st.title("📝 Розпізнавання та Переклад Рукописного Тексту")
st.markdown("---")

@st.cache_resource
def load_all_resources():
    model_path = "predict_crnn_model.h5"
    vocab_path = "correct_vocab.json"
    model = None
    num_to_char = {}

    # 1. Завантажуємо твою модель і словник
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
            num_to_char = {int(k): v for k, v in raw_vocab.items()}
    else:
        st.error(f"Файл словника {vocab_path} не знайдено!")

    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path, compile=False)
    else:
        st.error(f"Файл моделі {model_path} не знайдено!")

    # 2. Окремо створюємо і кешуємо EasyOCR Reader
    # Streamlit Cloud зрозуміє, що тут треба почекати
    ocr_reader = easyocr.Reader(["en"], gpu=False)

    return model, num_to_char, ocr_reader


with st.spinner('Завантаження моделі та словника... зачекайте...'):
    model, num_to_char, reader = load_all_resources()

# --- БІЧНА ПАНЕЛЬ (SIDEBAR) ---
st.sidebar.header("Налаштування")
languages = {
    "Англійська (English)": "en",
    "Іспанська (Español) - Незабаром": "es",
    "Українська - Незабаром": "uk",
    "Російська (Родной) - Незабаром": "ru"
}

selected_lang_name = st.sidebar.selectbox("Виберіть мову розпізнавання:", list(languages.keys()))
selected_lang_code = languages[selected_lang_name]

if selected_lang_code != "en":
    st.sidebar.warning(f"На жаль, мова '{selected_lang_name}' поки що недоступна. Модель навчена тільки на англійському рукопису. Буде використано Англійську.")
    selected_lang_code = "en"

# --- БЛОК ВИБОРУ СПОСОБУ ВВЕДЕННЯ ---
st.subheader("1. Оберіть спосіб введення")
input_method = st.radio(
    "Як ви хочете надати зображення?",
    ("Завантажити файл з пристрою", "Зробити фото камерою")
)
image_file = None

if input_method == "Зробити фото камерою":
    camera_capture = st.camera_input("Натисніть кнопку, щоб зробити знімок")
    if camera_capture is not None:
        image_file = camera_capture
        st.success("Фото успішно зроблено!")

elif input_method == "Завантажити файл з пристрою":
    uploaded_file = st.file_uploader(
        "Оберіть зображення рукописного тексту",
        type=['png', 'jpg', 'jpeg']
    )
    if uploaded_file is not None:
        image_file = uploaded_file
        st.success("Файл успішно завантажено!")
        st.image(uploaded_file, caption="Завантажене зображення", use_column_width=True)

# --- УНІФІКОВАНИЙ БЛОК ОБРОБКИ ЗОБРАЖЕННЯ ---
processed_text = ""
temp_filename = "temp_capture.png"

if image_file is not None:
    pil_image = Image.open(image_file)
    if pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")

    temp_original = "temp_capture.png"
    temp_cropped = "temp_cropped.png"
    pil_image.save(temp_original)

    st.subheader("2. Результат розпізнавання")
    if model is not None:
        with st.spinner("Виявлення та вирізання тексту..."):
            detection_success = detect_crop_text(
                temp_original, output_path=temp_cropped
            )

        if detection_success:
            with st.spinner("Розпізнавання тексту..."):
                cropped_pil = Image.open(temp_cropped)
                st.image(
                    cropped_pil,
                    caption="Вирізане слово для розпізнавання",
                    use_container_width=True,
                )

                img = processing_image(temp_cropped, image_size=(256, 64))
                img_batch = np.expand_dims(img, axis=0)

                preds = model.predict(img_batch)
                raw_text = ctc_best_path_decoding(preds[0], num_to_char)
                final_result = get_final_word(raw_text)
                os.remove(temp_original)
                os.remove(temp_cropped)

                st.text_area(
                    "Результат моделі (сирий):", value=final_result, height=70
                )

                st.info("Ви можете скопіювати текст з полів вище.")

                st.write("Чи правильна була відповідь асистента?")

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("Так, розпізнано вірно", use_container_width=True):
                        st.session_state.feedback = "yes"

                with col2:
                    if st.button("Ні, є помилки", use_container_width=True):
                        st.session_state.feedback = "no"

                if "feedback" in st.session_state:
                    if st.session_state.feedback == "yes":
                        st.success("ПЕРЕМОГА НАХУЙ")
                    elif st.session_state.feedback == "no":
                        st.error("сука еблан модель")

                        user_correction = st.text_input("Введіть правильний варіант слова (за бажанням):")
                        if user_correction:
                            st.success(f"Дякуємо! Ми врахуємо, що правильно писати: **{user_correction}**")
        else:
            st.warning(
                "Камера не змогла чітко знайти текст. Спробуйте зробити фото ближче або з кращим освітленням."
            )
            os.remove(temp_original)

st.markdown("---")
with st.expander("Як це працює?"):
    st.write("""
    Цей додаток використовує нейромережу архітектури CRNN (CNN + LSTM + CTC), навчену на датасеті IAM.
    Він вміє розпізнавати окремі англійські слова. Для кращого результату:
    - Тримайте камеру рівно.
    - Забезпечте гарне освітлення.
    - Намагайтеся, щоб в кадр потрапляло тільки одне слово.
    """)