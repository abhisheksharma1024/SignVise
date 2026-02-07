import customtkinter as ctk
import threading
import os
import sys
import importlib
from PIL import Image, ImageTk
import tkinter.messagebox as messagebox
import sys
import os

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Force the temporary EXE folder into the system path
sys.path.append(resource_path("."))

# ================= PATH HELPER =================
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# IMPORTANT: Ensure the script directory is in the system path for imports
script_dir = resource_path(".")
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# ================= COLORS =================
PRIMARY_BLUE = "#0A1A2F"
ACCENT_BLUE = "#1F6ED4"
GLOW_BLUE = "#4DA8FF"
BUTTON_BG = "#102A43"
BUTTON_HOVER = "#1C4E80"
POPUP_BG = "#081826"

class PragyaanApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # -------- WINDOW CONFIG --------
        self.title("Pragyaan")
        self.geometry("1100x700")
        self.minsize(1000, 650)
        ctk.set_appearance_mode("dark")
        self.configure(fg_color=PRIMARY_BLUE)

        self.main_container = ctk.CTkFrame(self, fg_color=PRIMARY_BLUE)
        self.main_container.pack(fill="both", expand=True)

        # -------- CENTER FRAME --------
        self.center_frame = ctk.CTkFrame(self.main_container, fg_color=PRIMARY_BLUE)
        self.center_frame.place(relx=0.5, rely=0.5, anchor="center")

        # -------- TITLE --------
        self.title_label = ctk.CTkLabel(
            self.center_frame,
            text="SignVise",
            font=("Carnivalee Freakshow", 300, "bold"),
            text_color=GLOW_BLUE
        )
        self.title_label.pack(pady=(0, 30))

        # -------- BUTTON GRID --------
        self.btn_frame = ctk.CTkFrame(self.center_frame, fg_color=PRIMARY_BLUE)
        self.btn_frame.pack()
        self.create_center_buttons()

        # -------- CHATBOT FLOAT BUTTON --------
        self.toggle_btn = ctk.CTkButton(
            self, text="🤖", width=60, height=60, corner_radius=30,
            fg_color=ACCENT_BLUE, hover_color=GLOW_BLUE,
            font=("Arial", 22), command=self.toggle_popup
        )
        self.toggle_btn.place(relx=0.95, rely=0.92, anchor="center")
        self.popup = None

    def create_center_buttons(self):
        buttons = [
            ("👋 VIRTUAL MOUSE", "Virtual Mouse using Hand Gestures", self.load_hatho_ka_jadu),
            ("🖖 HAND MASTER", "Gesture Based Game Control", self.load_hand_master),
            ("👽 TEXT TO SPEECH", "Gesture to Speech Translator", self.load_harhit),
            ("🚨 SOS", "Emergency Gesture SOS System", self.load_sos),
        ]

        for i, (title, desc, cmd) in enumerate(buttons):
            btn = ctk.CTkButton(
                self.btn_frame, 
                text=title, 
                width=340, 
                height=80, 
                corner_radius=40,
                fg_color=BUTTON_BG, 
                hover_color=GLOW_BLUE, 
                border_width=2,
                border_color=ACCENT_BLUE, 
                font=("Buffalo Inline 2 Grunge", 22, "bold"),
                command=lambda t=title, d=desc, c=cmd: self.open_feature_popup(t, d, c)
            )
            btn.grid(row=i // 2, column=i % 2, padx=40, pady=28)

    def open_feature_popup(self, title, desc, command):
        popup = ctk.CTkToplevel(self)
        popup.geometry("400x280")
        popup.attributes("-topmost", True) # Ensure it stays on top
        popup.configure(fg_color=POPUP_BG)

        ctk.CTkLabel(popup, text=title, font=("Interea", 30, "bold"), text_color=GLOW_BLUE).pack(pady=20)
        ctk.CTkLabel(popup, text=desc, wraplength=340, font=("Cream Cake", 23), text_color="lightgray").pack(pady=10)
        ctk.CTkButton(popup, text="🚀 Launch Module", command=lambda: [popup.destroy(), command()]).pack(pady=25)

    def toggle_popup(self):
        if self.popup and self.popup.winfo_exists():
            self.popup.destroy()
            self.popup = None
        else:
            self.show_chatbot_menu()

    def show_chatbot_menu(self):
        self.popup = ctk.CTkToplevel(self)
        self.popup.geometry("320x300+760+420")
        self.popup.overrideredirect(True)
        self.popup.attributes("-topmost", True)
        self.popup.configure(fg_color=POPUP_BG)

        ctk.CTkLabel(self.popup, text="PROMPTYUSH", font=("Buffalo Inline 2 Grunge", 18, "bold"), text_color=GLOW_BLUE).pack(pady=20)
        ctk.CTkButton(self.popup, text="💬 Open Chatbot", command=self.load_promptyush).pack(pady=15)
        ctk.CTkButton(self.popup, text="❌ Close", fg_color="#8B0000", command=self.popup.destroy).pack(pady=20)

    # ========== MODULE LOADERS (DIAGNOSTIC ENABLED) ==========
    def _exec_mod(self, mod_name, func_name):
        try:
            # Re-verify pathing inside thread
            curr_path = resource_path(".")
            if curr_path not in sys.path:
                sys.path.insert(0, curr_path)
            
            # Dynamic import
            module = importlib.import_module(mod_name)
            importlib.reload(module)
            
            # Execute the function
            run_func = getattr(module, func_name)
            run_func(self) 
        except Exception as e:
            # If it doesn't open, this will tell us WHY
            messagebox.showerror("Module Error", f"Failed to launch {mod_name}:\n{str(e)}")

    def load_hatho_ka_jadu(self):
        threading.Thread(target=lambda: self._exec_mod("VIRTUAL_MOUSE_CODE", "run_virtual_mouse"), daemon=True).start()

    def load_hand_master(self):
        threading.Thread(target=lambda: self._exec_mod("GAME_CODE", "run_hand_master"), daemon=True).start()

    def load_harhit(self):
        threading.Thread(target=lambda: self._exec_mod("GTTSM_MODEL_RUN_CODE", "run_gttsm_model"), daemon=True).start()

    def load_sos(self):
        threading.Thread(target=lambda: self._exec_mod("EMERGENCY_SOS_CODE", "run_emergency_sos"), daemon=True).start()

    def load_promptyush(self):
        threading.Thread(target=lambda: self._exec_mod("CHATBOT", "run_promptyush"), daemon=True).start()

if __name__ == "__main__":
    app = PragyaanApp()
    app.mainloop()