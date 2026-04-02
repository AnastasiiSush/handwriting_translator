import numpy as np
import cv2
import tensorflow as tf
from data_utils import processing_image, load_iam_data, create_vocabluary
import random
import os
from spellchecker import SpellChecker

spell = SpellChecker()

def ctc_best_path_decoding(y_pred, num_to_char):
    best_path = np.argmax(y_pred, axis=-1)
    collapsed_path = []

    for i in range(len(best_path)):
        if i == 0 or best_path[i] != best_path[i - 1]:
            collapsed_path.append(best_path[i])

    final_text = ""
    for idx in collapsed_path:
        if idx != 0:
            char_data = num_to_char.get(idx) or num_to_char.get(str(idx), "")

            if char_data:
                if isinstance(char_data, (list, tuple)):
                    if len(char_data) >= 2:
                        final_text += char_data[1]
                elif isinstance(char_data, str):
                    final_text += char_data

    return final_text

def predict_single_image(model, image_path, num_to_char, image_size=(256,64)):
    img = processing_image(image_path, image_size=image_size)
    img_batch = np.expand_dims(img,axis=0)
    preds = model.predict(img_batch)
    decoded_text = ctc_best_path_decoding(preds[0], num_to_char)
    return decoded_text

def get_final_word(raw_prediction):
    raw_prediction = raw_prediction.strip()
    corrected = spell.correction(raw_prediction)
    if corrected is None:
        return raw_prediction

    return corrected

if __name__ == "__main__":
    print("Завантаження словника...")
    DATA_DIR = "./iam_words/"
    WORDS_TXT_PATH = os.path.join(DATA_DIR, "words.txt")
    IMG_DIR = os.path.join(DATA_DIR, "words")
    paths, texts = load_iam_data(WORDS_TXT_PATH, IMG_DIR)
    char_to_num, num_to_char, vocab_size = create_vocabluary(texts)

    print("Завантаження моделі...")
    model_path = "predict_crnn_model.h5"

    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Модель успішно завантажена!")

        random_index = random.randint(0, len(paths)- 1)
        test_img = paths[random_index]
        true_label = texts[random_index]
        predicted_label = predict_single_image(model, test_img, num_to_char)
        final_result = get_final_word(predicted_label)

        print("\n=== РЕЗУЛЬТАТ ВИПАДКОВОГО ТЕСТУ ===")
        print(f"Індекс картинки: {25}")
        print(f"Шлях до файлу: {test_img}")
        print(f"Реальне слово (Ground Truth): {true_label}")
        print(f"Розпізнано моделлю (Prediction): {predicted_label}")
        print(f"Автокорректор (Prediction): {final_result}")


        img_to_show = cv2.imread(test_img)
        cv2.imshow(f"Real word: {true_label} | AI: {predicted_label}", img_to_show)
        print("\nНатисни будь-яку клавішу у вікні з картинкою, щоб закрити її.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print(f"Помилка: Файл {model_path} не знайдено!")