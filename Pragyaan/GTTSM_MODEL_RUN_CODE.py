import cv2
import mediapipe as mp
import time
import numpy as np
from collections import deque
from spellchecker import SpellChecker
from gtts import gTTS
from playsound import playsound
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os
import sys
from spellchecker import SpellChecker

# --- EXE PATH HANDLING ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    return os.path.join(base_path, relative_path)

# --- WRAPPER FOR DASHBOARD ---
def run_gttsm_model(self=None):
    # We move the initialization inside so it only runs when the button is clicked
    
    # Load model data using resource_path
    try:
        X = np.load(resource_path(os.path.join("ASSETS", "GTTSM_Model_Assets", "landmarks.npy")))
        y = np.load(resource_path(os.path.join("ASSETS", "GTTSM_Model_Assets", "labels.npy")))
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    def normalize_landmarks(sample):
        sample = sample.reshape(-1, 3)
        wrist = sample[0]
        sample = sample - wrist
        norm = np.linalg.norm(sample)
        if norm > 0:
            sample = sample / norm
        return sample.flatten()

    X_norm = np.array([normalize_landmarks(s) for s in X])

    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=42
    )

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    print("✅ Model trained with accuracy:", knn.score(X_test, y_test))

    # -----------------------------
    # Initialize Mediapipe & helpers
    # -----------------------------
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    # --- UPDATED SPELLCHECKER INITIALIZATION ---
    try:
        # We look for the dictionary file within the collected site-packages in the EXE
        dictionary_path = resource_path(os.path.join("spellchecker", "resources", "en.json.gz"))
        if os.path.exists(dictionary_path):
            spell = SpellChecker(language=None, local_dictionary=dictionary_path)
        else:
            # Fallback for development mode
            spell = SpellChecker(language='en')
    except Exception as e:
        print(f"⚠️ Spellchecker warning: {e}. Falling back to default.")
        spell = SpellChecker(language='en')

    stable_buffer = deque(maxlen=15)
    # Using non-global list to keep it contained within the function
    sub_lines = ["", "", ""]

    last_space_time = time.time()
    last_letter_time = time.time()
    last_gesture_time = time.time()

    # -----------------------------
    # Helper functions (Nested)
    # -----------------------------
    def classify_hand(landmarks):
        features = []
        for lm in landmarks:
            features.extend([lm.x, lm.y, lm.z])
        features = np.array(features)
        features = normalize_landmarks(features)
        return knn.predict([features])[0]

    def update_subtitles(text, lines):
        lines[-1] += text
        if len(lines[-1]) > 60:
            lines.append("")
            return lines[-3:]
        return lines

    def autocorrect_last_word(lines):
        words = lines[-1].split()
        if words:
            last_word = words[-1]
            if len(last_word) >= 3:
                corrected = spell.correction(last_word.lower())
                if corrected:
                    words[-1] = corrected.capitalize()
                    lines[-1] = " ".join(words)
        return lines

    def speak_word(word):
        if not word.strip():
            return
        try:
            tts = gTTS(text=word, lang='en', tld='co.uk')
            # Unique filename to prevent permission errors in EXE
            filename = f"temp_{int(time.time())}.mp3"
            tts.save(filename)
            playsound(filename)
            if os.path.exists(filename):
                os.remove(filename)
        except Exception as e:
            print(f"Audio Error: {e}")

    # -----------------------------
    # Live Capture Loop
    # -----------------------------
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=2
    ) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            current_letter = None

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    landmarks = hand_landmarks.landmark

                    if handedness.classification[0].label == "Right":
                        for lm in landmarks:
                            lm.x = 1.0 - lm.x

                    # Predict letter
                    letter = classify_hand(landmarks)
                    stable_buffer.append(letter)

                    # Stable prediction
                    if stable_buffer.count(letter) > len(stable_buffer) // 2:
                        current_letter = letter
                        if time.time() - last_letter_time > 3:  # cooldown
                            sub_lines = update_subtitles(current_letter, sub_lines)
                            last_letter_time = time.time()
                        last_gesture_time = time.time()

            # ✅ Add space after 3s inactivity
            if time.time() - last_gesture_time > 3 and (time.time() - last_space_time > 3):
                sub_lines = update_subtitles(" ", sub_lines)
                sub_lines = autocorrect_last_word(sub_lines)
                words = sub_lines[-1].split()
                if words:
                    speak_word(words[-1])
                last_space_time = time.time()

            # ✅ Reset subtitles after 6s inactivity
            if time.time() - last_gesture_time > 6:
                sub_lines = ["", "", ""]

            # Draw subtitles with black background
            h, w, _ = frame.shape
            y_offset = h - 60
            for i, line in enumerate(sub_lines[-3:]):
                if line.strip():
                    (text_w, text_h), _ = cv2.getTextSize(
                        line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )
                    cv2.rectangle(
                        frame,
                        (45, y_offset + i*20 - text_h - 5),
                        (55 + text_w, y_offset + i*20 + 5),
                        (0, 0, 0),
                        -1
                    )
                    cv2.putText(
                        frame,
                        line,
                        (50, y_offset + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

            cv2.imshow("ASL to Text & Speech", frame)

            if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
                break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# RUN STANDALONE OR VIA IMPORT
# -----------------------------
if __name__ == "__main__":
    run_gttsm_model()