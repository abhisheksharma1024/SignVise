import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm   # optional, for progress bars

mp_hands = mp.solutions.hands

data = []
labels = []

# ✅ Update this path to your dataset folder
base_dir = r"C:\\Users\\hshar\\Desktop\\PRAGYAAN AI MODEL\\ASL_DATASET\\asl_alphabet_train\\asl_alphabet_train"

with mp_hands.Hands(static_image_mode=True) as hands:
    for label in tqdm(os.listdir(base_dir), desc="Classes"):
        folder = os.path.join(base_dir, label)
        if not os.path.isdir(folder):
            continue

        for img_file in tqdm(os.listdir(folder), desc=f"Processing {label}", leave=False):
            img_path = os.path.join(folder, img_file)
            image = cv2.imread(img_path)
            if image is None:
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    data.append(landmarks)
                    labels.append(label)

print("✅ Dataset loaded:", len(data), "samples across", len(set(labels)), "classes")

# Save dataset
np.save("landmarks.npy", np.array(data))
np.save("labels.npy", np.array(labels))
print("💾 Saved dataset to landmarks.npy and labels.npy")



