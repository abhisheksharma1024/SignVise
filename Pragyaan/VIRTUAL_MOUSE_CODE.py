import cv2
import mediapipe as mp
import pyautogui
import time
import math
import os
import sys
import ctypes
# This tells Windows to treat our app as 'High DPI Aware' 
# This makes the mouse coordinates 100% pixel-perfect
ctypes.windll.shcore.SetProcessDpiAwareness(1)

# ---------------- EXE RESOURCE PATH HELPER ----------------
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ---------------- CONFIG ----------------
CLICK_DIST = 25
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0  # CRITICAL: Removes lag between commands

# ---------------- UTILS ----------------
def calculate_dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def get_coords(landmarks, w, h):
    return {
        i: (int(landmarks[i].x * w), int(landmarks[i].y * h))
        for i in range(len(landmarks))
    }

# ---------------- MAIN WRAPPER ----------------
def run_virtual_mouse(self=None):
    """ This function is called by your Main Dashboard """
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Use 640x480 for the best balance of speed and accuracy
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mp_hands = mp.solutions.hands
    hand_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    screen_w, screen_h = pyautogui.size()
    plocx, plocy = 0, 0
    click_time = 0
    left_dragging = False
    
    # Accuracy Tweak: Lower = faster, Higher = smoother
    SMOOTHING = 4  
    
    prev_time = time.time()

    print("🚀 Full Virtual Mouse Active. Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hand_detector.process(rgb_frame)

        if results and results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]
            coords = get_coords(hand_lms.landmark, w, h)

            # --- 1. MOUSE MOVEMENT ---
            index_tip = coords[8]
            raw_x = index_tip[0] * screen_w / w
            raw_y = index_tip[1] * screen_h / h

            # Interpolation for fluid movement
            clocx = plocx + (raw_x - plocx) / SMOOTHING
            clocy = plocy + (raw_y - plocy) / SMOOTHING

            pyautogui.moveTo(clocx, clocy, _pause=False)
            plocx, plocy = clocx, clocy

            # --- 2. GESTURE POINTS ---
            thumb = coords[4]
            index = coords[8]
            middle = coords[12]
            ring = coords[16]
            wrist = coords[0]

            dist_index = calculate_dist(index, thumb)
            dist_middle = calculate_dist(middle, thumb)
            dist_ring = calculate_dist(ring, thumb)

            # Visual Feedback
            for lm in [4, 8, 12, 16, 0]:
                cv2.circle(frame, coords[lm], 6, (255, 255, 0), -1)

            # --- 3. CLICK LOGIC ---
            # Left Click (Index + Thumb)
            if dist_index < CLICK_DIST and time.time() - click_time > 0.3:
                pyautogui.click()
                click_time = time.time()
                cv2.circle(frame, index, 15, (0, 255, 0), cv2.FILLED)

            # Right Click (Middle + Thumb)
            elif dist_middle < CLICK_DIST and time.time() - click_time > 0.5:
                pyautogui.rightClick()
                click_time = time.time()
                cv2.circle(frame, middle, 15, (0, 0, 255), cv2.FILLED)

            # Dragging (Ring + Thumb)
            if dist_ring < CLICK_DIST:
                if not left_dragging:
                    pyautogui.mouseDown()
                    left_dragging = True
            else:
                if left_dragging:
                    pyautogui.mouseUp()
                    left_dragging = False

            # --- 4. SCROLL LOGIC (RESTORED) ---
            # Only scroll if fingers are not in a 'click' position
            if dist_index > CLICK_DIST + 10 and dist_middle > CLICK_DIST + 10:
                # Scroll UP (Thumb well above wrist)
                if thumb[1] < wrist[1] - 60:
                    pyautogui.scroll(100)
                    cv2.putText(frame, "SCROLL UP", (w-150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # Scroll DOWN (Thumb below or near wrist)
                elif thumb[1] > wrist[1] + 40:
                    pyautogui.scroll(-100)
                    cv2.putText(frame, "SCROLL DOWN", (w-150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # FPS & UI
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time))
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("PRAGYAAN Virtual Mouse", frame)

        if cv2.waitKey(1) & 0xFF == 27: # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_virtual_mouse()