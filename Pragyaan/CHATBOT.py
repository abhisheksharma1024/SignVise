import tkinter as tk
from tkinter import scrolledtext
import threading
import time
import sys
import os
import speech_recognition as sr
import pyttsx3
import importlib.util

# --- EXE RESOURCE PATH HELPER ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ---------------- CONFIG ----------------
BOT_NAME = "Promptyush"
DASHBOARD_BLUE = "#0A2A66"
CHAT_BG = "#E9EEF6"
USER_BG = "#CFE9FF"
BOT_BG = "#FFFFFF"

# ---------------- CHATBOT ENGINE ----------------
class PromptyushBot:
    def __init__(self, root):
        self.root = root
        self.root.title(BOT_NAME)
        self.root.geometry("460x600")
        self.root.configure(bg=CHAT_BG)
        
        # --- AUDIO LOCK ---
        # Prevents Speak and Listen from crashing each other
        self.audio_lock = threading.Lock()
        
        # Initialize TTS Engine safely
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 165)
        except:
            self.engine = None

        # Load Intents Safely
        self.get_response = self.load_intents()

        self.create_dashboard()
        self.create_chat_area()
        self.create_input_area()

        initial_greeting = "Hello, I am Promptyush.\nHow can I help you?"
        self.bot_message(initial_greeting)
        self.speak(initial_greeting)

    def load_intents(self):
        """ Dynamically loads intents.py from the ASSETS folder """
        try:
            intents_path = resource_path(os.path.join("ASSETS", "CHATBOT_ASSETS", "intents.py"))
            spec = importlib.util.spec_from_file_location("intents", intents_path)
            intents_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(intents_module)
            return intents_module.get_response
        except Exception as e:
            print(f"Chatbot Load Error: {e}")
            return lambda x: "I'm having trouble connecting to my brain (intents.py)."

    def speak(self, text):
        if self.engine:
            def _speak():
                with self.audio_lock: # Ensure we don't listen while speaking
                    try:
                        self.engine.say(text)
                        self.engine.runAndWait()
                    except:
                        pass 
            threading.Thread(target=_speak, daemon=True).start()

    def create_dashboard(self):
        dash = tk.Frame(self.root, bg=DASHBOARD_BLUE, height=60)
        dash.pack(fill="x")
        tk.Label(dash, text=BOT_NAME, fg="white", bg=DASHBOARD_BLUE, 
                 font=("Segoe UI", 16, "bold")).pack(side="left", padx=15)
        tk.Button(dash, text="Clear", command=self.clear_chat, bg=DASHBOARD_BLUE, 
                  fg="white", border=0, font=("Segoe UI", 11)).pack(side="right", padx=10)

    def create_chat_area(self):
        self.chat = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, font=("Segoe UI", 10),
                                             bg=CHAT_BG, state="disabled")
        self.chat.pack(fill="both", expand=True, padx=8, pady=8)

    def create_input_area(self):
        bottom = tk.Frame(self.root, bg=CHAT_BG)
        bottom.pack(fill="x", pady=6)
        self.entry = tk.Entry(bottom, font=("Segoe UI", 11), width=26)
        self.entry.pack(side="right", padx=6)
        self.entry.bind("<Return>", self.send_text)
        tk.Button(bottom, text="➤", command=self.send_text, border=0).pack(side="right", padx=4)
        tk.Button(bottom, text="🎤", command=self.voice_input, border=0).pack(side="right")

    def bot_message(self, msg):
        self.chat.config(state="normal")
        self.chat.insert(tk.END, f"\n{BOT_NAME}: {msg}\n", "bot")
        self.chat.tag_config("bot", justify="left", background=BOT_BG)
        self.chat.config(state="disabled")
        self.chat.yview(tk.END)

    def user_message(self, msg, spoken=False):
        self.chat.config(state="normal")
        self.chat.insert(tk.END, f"\nYou: {msg} {'🎤' if spoken else ''}\n", "user")
        self.chat.tag_config("user", justify="right", background=USER_BG)
        self.chat.config(state="disabled")
        self.chat.yview(tk.END)

    def send_text(self, event=None):
        text = self.entry.get().strip()
        if text:
            self.entry.delete(0, tk.END)
            self.user_message(text)
            threading.Thread(target=self.process_reply, args=(text,), daemon=True).start()

    def process_reply(self, text):
        reply = self.get_response(text)
        # Use after() to update UI from thread safely
        self.root.after(0, lambda: self.bot_message(reply))
        self.speak(reply)

    def voice_input(self):
        def listen():
            r = sr.Recognizer()
            # Dynamic energy helps in noisy rooms
            r.dynamic_energy_threshold = True 
            
            try:
                with sr.Microphone() as source:
                    # Thread-safe UI update
                    self.root.after(0, lambda: self.bot_message("Listening... (Speak now)"))
                    
                    # Phrase time limit prevents the mic from hanging on background noise
                    audio = r.listen(source, timeout=5, phrase_time_limit=5)
                
                self.root.after(0, lambda: self.bot_message("Processing voice..."))
                text = r.recognize_google(audio)
                
                # Push back to main thread
                self.root.after(0, lambda: self.user_message(text, spoken=True))
                self.root.after(0, lambda: self.process_reply(text))
                
            except sr.WaitTimeoutError:
                self.root.after(0, lambda: self.bot_message("No speech detected."))
            except sr.UnknownValueError:
                self.root.after(0, lambda: self.bot_message("Sorry, I didn't catch that."))
            except Exception as e:
                self.root.after(0, lambda: self.bot_message("Mic error. Please try again."))
                print(f"Voice Error: {e}")

        threading.Thread(target=listen, daemon=True).start()

    def clear_chat(self):
        self.chat.config(state="normal")
        self.chat.delete("1.0", tk.END)
        self.chat.config(state="disabled")

# ---------------- DASHBOARD INTERFACE ----------------
def run_promptyush(self=None):
    """ Call this function from your dashboard button """
    chat_win = tk.Toplevel()
    PromptyushBot(chat_win)

# Standalone testing
if __name__ == "__main__":
    main_root = tk.Tk()
    tk.Button(main_root, text="Open Chatbot", command=lambda: run_promptyush()).pack(pady=20)
    main_root.mainloop()