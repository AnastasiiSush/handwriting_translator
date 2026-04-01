import streamlit as st
import cv2
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from streamlit import camera_input

from data_utils import processing_image, load_iam_data, create_vocabluary
from predict import ctc_best_path_decoding

st.set_page_config(page_title="Handwritten Text Recognizer", page_icon="📝")
st.title("📝 Розпізнавання та Переклад Рукописного Тексту")
st.markdown("---")

@st.cache_resource
def load_all_resources():
    DATA_DIR = "./iam_words/"
    WORDS_TXT_PATH = os.path.join(DATA_DIR, "words.txt")
    IMG_DIR = os.path.join(DATA_DIR, "words")
    if os.path.exists(WORDS_TXT_PATH):
        paths, texts = load_iam_data(WORDS_TXT_PATH, IMG_DIR)
        char_to_num, num_to_char, vocab_size = create_vocabluary(texts)
    else:
        st.warning("Дані IAM не знайдені поруч. Використовується обмежений словник.")
        characters = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        char_to_num = {char: idx + 1 for idx, char in enumerate(characters)}
        num_to_char = {idx + 1: char for idx, char in enumerate(characters)}
        num_to_char[0] = "[blank]"

    model_path = "predict_crnn_model.h5"
    model = None
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path, compile=False)
    else:
        st.error(f"Файл моделі {model_path} не знайдено! Навчіть модель спочатку.")
    return model, num_to_char

with st.spinner('Завантаження моделі та словника... зачекайте...'):
    model, num_to_char = load_all_resources()

# --- БІЧНА ПАНЕЛЬ (SIDEBAR) ---
st.sidebar.header("Налаштування")

# 1. Вибір мови (твоє завдання)
languages = {
    "Англійська (English)": "en",
    "Іспанська (Español) - Незабаром": "es",
    "Українська - Незабаром": "uk",
    "Російська (Родной) - Незабаром": "ru"
}

selected_lang_name = st.sidebar.selectbox("Виберіть мову розпізнавання:", list(languages.keys()))
selected_lang_code = languages[selected_lang_name]

# Попередження, якщо вибрано не англійську
if selected_lang_code != "en":
    st.sidebar.warning(f"На жаль, мова '{selected_lang_name}' поки що недоступна. Модель навчена тільки на англійському рукопису. Буде використано Англійську.")
    selected_lang_code = "en" # Примусово ставимо англійську

# --- ГОЛОВНА ЗОНА ---
st.subheader("1. Зробіть фото тексту")
camera_input = st.camera_input("Натисніть кнопку, щоб зробити знімок")

processed_text = ""

if camera_input is not None:
    st.success("Фото успішно зроблено!")
    pil_image = Image.open(camera_input)
    temp_filename = "temp_capture.png"
    pil_image.save(temp_filename)

    st.subheader("2. Результат розпізнавання")
    if model is not None:
        with st.spinner('Розпізнавання тексту...'):
            img = processing_image(temp_filename, image_size=(256,64))
            img_batch = np.expand_dims(img, axis=0)

            preds = model.predict(img_batch)
            processed_text = ctc_best_path_decoding(preds[0], num_to_char)
            os.remove(temp_filename)
            st.text_area("Розпізнаний текст:", value=processed_text, height=100)

            st.info("Ви можете скопіювати текст з поля вище.")
    else:
        st.error("Неможливо виконати розпізнавання: модель не завантажена.")

st.markdown("---")
st.write("Тип даних словника:", type(num_to_char))
st.write("Приклад словника:", list(num_to_char.items())[:5])
with st.expander("Як це працює?"):
    st.write("""
    Цей додаток використовує нейромережу архітектури CRNN (CNN + LSTM + CTC), навчену на датасеті IAM.
    Він вміє розпізнавати окремі англійські слова. Для кращого результату:
    - Тримайте камеру рівно.
    - Забезпечте гарне освітлення.
    - Намагайтеся, щоб в кадр потрапляло тільки одне слово.
    """)