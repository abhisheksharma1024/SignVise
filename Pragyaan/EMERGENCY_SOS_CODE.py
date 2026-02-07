import cv2
import time
import mediapipe as mp
from twilio.rest import Client
import geocoder
import threading
import tkinter as tk
from tkinter import messagebox
import os
import sys

# --- EXE RESOURCE PATH HELPER ---
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ---------------- TWILIO CONFIG ----------------
# NOTE: Ensure these are active. For EXE, avoid sharing these publicly.
ACCOUNT_SID = "ACd5baf840fb439cb462cae8fd3987eaa2" 
AUTH_TOKEN = "da90ad524362b5ca74ebeb2b3160438f"
TWILIO_NUMBER = "+18314806263"
TARGET_NUMBER = "+918630606547"

# ---------------- SOS LOGIC WRAPPER ----------------
class SOSSystem:
    def __init__(self):
        self.running = False
        self.cap = None
        self.trigger_time = 5
        self.client = Client(ACCOUNT_SID, AUTH_TOKEN)
        
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def get_location(self):
        try:
            # Using a backup method for geocoder
            g = geocoder.ip('me')
            if g.ok and g.latlng:
                lat, lon = g.latlng
                return f"https://www.google.com/maps?q={lat},{lon}"
        except:
            pass
        return "Location not available (Check Internet)"

    def send_emergency_sms(self):
        def _send():
            try:
                location = self.get_location()
                message = (
                    "🚨 EMERGENCY ALERT 🚨\n"
                    "Closed fist detected for 5 seconds.\n"
                    f"📍 Location: {location}"
                )
                self.client.messages.create(
                    body=message,
                    from_=TWILIO_NUMBER,
                    to=TARGET_NUMBER
                )
                print("✅ Emergency SMS sent successfully")
            except Exception as e:
                print(f"❌ SMS Failed: {e}")

        # Send in background thread so camera doesn't freeze
        threading.Thread(target=_send, daemon=True).start()

    def is_closed_fist(self, hand_landmarks):
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        for tip, pip in zip(finger_tips, finger_pips):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                return False
        return True

    def sos_loop(self):
        self.cap = cv2.VideoCapture(0)
        fist_start_time = None
        sms_sent = False

        while self.running and self.cap.isOpened():
            success, frame = self.cap.read()
            if not success: break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    if self.is_closed_fist(hand_landmarks):
                        if fist_start_time is None:
                            fist_start_time = time.time()
                        
                        elapsed = int(time.time() - fist_start_time)
                        cv2.putText(frame, f"SOS TRIGGER IN: {self.trigger_time - elapsed}s", (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                        if elapsed >= self.trigger_time and not sms_sent:
                            self.send_emergency_sms()
                            sms_sent = True
                    else:
                        fist_start_time = None
                        sms_sent = False

            cv2.imshow("SOS Camera - Press ESC to Exit Feed", frame)
            if cv2.waitKey(1) & 0xFF == 27: break

        if self.cap: self.cap.release()
        cv2.destroyAllWindows()
        self.running = False

# ---------------- EXTERNAL CALL ----------------
def run_emergency_sos(parent_window=None):
    sos = SOSSystem()
    
    # Use Toplevel so it doesn't kill the main dashboard
    win = tk.Toplevel()
    win.title("PRAGYAAN SOS")
    win.geometry("350x250")
    win.configure(bg="#1a1a1a")
    win.attributes("-topmost", True) # Keep on top

    tk.Label(win, text="🚨 EMERGENCY SOS", fg="#ff4d4d", bg="#1a1a1a", font=("Arial", 16, "bold")).pack(pady=20)

    def start():
        if not sos.running:
            sos.running = True
            threading.Thread(target=sos_loop_wrapper, daemon=True).start()
            btn_start.config(state="disabled", bg="gray")

    def sos_loop_wrapper():
        sos.sos_loop()
        # Reset button when loop ends
        try: btn_start.config(state="normal", bg="#28a745")
        except: pass

    btn_start = tk.Button(win, text="▶ ACTIVATE SOS", bg="#28a745", fg="white", font=("Arial", 12, "bold"),
                          command=start, width=20, height=2)
    btn_start.pack(pady=10)

    tk.Button(win, text="⛔ CLOSE", bg="#dc3545", fg="white", font=("Arial", 10),
              command=win.destroy, width=15).pack(pady=10)

# ---------------- STANDALONE RUN ----------------
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw() # Hide the empty main window
    run_emergency_sos()
    root.mainloop()