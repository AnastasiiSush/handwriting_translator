import os
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
import random

def detect_crop_text(image_path, output_path="temp_cropped.png"):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mser = cv2.MSER_create(_delta=5, _min_area=60, _max_area=14400)
        regions, _ = mser.detectRegions(gray)

        if not regions:
            return False

        all_points = []
        for region in regions:
            for point in region:
                all_points.append(point)

        if not all_points:
            return False

        all_points = np.array(all_points)
        xmin, ymin = np.min(all_points, axis=0)
        xmax, ymax = np.max(all_points, axis=0)

        h, w, _ = img.shape
        h_pad = int((ymax - ymin) * 0.2)
        w_pad = int((xmax - xmin) * 0.2)

        ymin = int(max(0, ymin - h_pad))
        ymax = int(min(h, ymax + h_pad))
        xmin = int(max(0, xmin - w_pad))
        xmax = int(min(w, xmax + w_pad))

        cropped_img = img[ymin:ymax, xmin:xmax]

        if cropped_img.size == 0:
            return False

        cv2.imwrite(output_path, cropped_img)
        return True

    except Exception as e:
        print(f"Помилка детекції: {e}")
        return False


def load_iam_data(txt_path, img_dir):
    image_paths = []
    labels = []
    with open(txt_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ")
            if len(parts) < 9:
                continue
            if parts[1] != "ok":
                continue

            word_id = parts[0]
            label = parts[-1]
            id_parts = word_id.split("-")
            subfolder_1 = id_parts[0]
            subfolder_2 = f"{id_parts[0]}-{id_parts[1]}"
            img_path = os.path.join(img_dir, subfolder_1, subfolder_2, f"{word_id}.png")
            if os.path.exists(img_path) and len(label) > 0:
                image_paths.append(img_path)
                labels.append(label)
    return image_paths, labels


# 2. Створення словника
def create_vocabluary(labels):
    characters = set()
    for label in labels:
        for char in label:
            characters.add(char)
    characters = sorted(list(characters))
    char_to_num = {char: idx + 1 for idx, char in enumerate(characters)}
    num_to_char = {idx + 1: char for idx, char in enumerate(characters)}
    char_to_num["[blank]"] = 0
    num_to_char[0] = "[blank]"
    return char_to_num, num_to_char, len(characters) + 1


# 3. Кодування тексту
def encode_text(text, char_to_num, max_len=32, vocab_size=80):
    encoded = np.zeros(max_len, dtype=np.int32)
    for i, char in enumerate(text[:max_len]):
        if char in char_to_num:
            idx = char_to_num[char]
            if idx < vocab_size - 1:
                encoded[i] = idx
    return encoded


# 4. Обробка зображення (з виправленою помилкою!)
def processing_image(image_path, image_size=(256, 64)):
    target_w, target_h = image_size
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((target_h, target_w, 1), dtype=np.float32)

    img_blurred = cv2.medianBlur(img, 3)
    img_thresh = cv2.adaptiveThreshold(
        img_blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    coords = np.column_stack(np.where(img_thresh == 0))

    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) > 0.5:
            (h, w) = img_thresh.shape
            center = (w // 2, h // 2)

            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            img_thresh = cv2.warpAffine(
                img_thresh, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255
            )

    h, w = img_thresh.shape
    scale = target_h / h
    new_w = int(w * scale)
    if new_w > target_w:
        new_w = target_w
    image_resized = cv2.resize(img_thresh, (new_w, target_h))
    final_image = np.ones((target_h, target_w), dtype=np.float32) * 255
    final_image[:, :new_w] = image_resized
    final_image = 1.0 - (final_image / 255.0)
    final_image = np.expand_dims(final_image, axis=-1)
    return final_image


#5. Class
class IAMDataGenerator(Sequence):
    def __init__(self, image_paths, labels, char_to_num, batch_size=32, image_size=(256,64), max_text_len=32, shuffle=True, augment=False):
        super().__init__()
        self.image_paths = image_paths
        self.labels = labels
        self.char_to_num = char_to_num
        self.batch_size = batch_size
        self.image_size = image_size
        self.max_text_len = max_text_len
        self.augment = augment
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.labels))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = np.zeros((self.batch_size, self.image_size[1], self.image_size[0], 1), dtype=np.float32)
        batch_labels = np.zeros((self.batch_size, self.max_text_len), dtype=np.int32)
        input_length = np.ones((self.batch_size, 1), dtype=np.int32) * (self.image_size[0] // 4)
        label_length = np.zeros((self.batch_size, 1), dtype=np.int32)

        for i, idx in enumerate(indexes):
            img_processed = processing_image(self.image_paths[idx], image_size=self.image_size)
            img = np.squeeze(img_processed)

            if self.augment:
                if random.random() < 0.5:
                    angle = random.uniform(-3, 3)
                    h, w = img.shape
                    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

                if random.random() < 0.3:
                    kernel_size = random.choice([2, 3])
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    if random.random() < 0.5:
                        img = cv2.dilate(img, kernel, iterations=1)
                    else:
                        img = cv2.erode(img, kernel, iterations=1)

                if random.random() < 0.2:
                    sigma = random.uniform(0.1, 1.0)
                    img = cv2.GaussianBlur(img, (3, 3), sigma)

            img = np.clip(img, 0, 1)
            batch_images[i] = np.expand_dims(img, axis=-1)
            text = self.labels[idx]
            vocab_sz = len(self.char_to_num)

            batch_labels[i] = encode_text(text, self.char_to_num, max_len=self.max_text_len, vocab_size=vocab_sz)
            label_length[i][0] = max(1, min(len(text), self.max_text_len))



        inputs = {
            "image": batch_images,
            "label": batch_labels,
            "input_length": input_length,
            "label_length": label_length
        }
        return inputs, np.zeros((self.batch_size))
