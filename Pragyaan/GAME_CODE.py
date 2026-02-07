import cv2
import mediapipe as mp
import pygame
import numpy as np
import time
import os
import pickle
import json
import random
import sys

# --- EXE RESOURCE PATH HELPER ---
# This is the bridge that tells the EXE to look in the temporary folder
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def run_hand_master(self=None):
    # --- PATH CONFIGURATION (Standardized for ASSETS folder) ---
    ASSETS_DIR = resource_path(os.path.join("ASSETS", "LEARNING_GAME", "assets"))
    MODEL_PATH = resource_path(os.path.join("ASSETS", "LEARNING_GAME", "sign_model.pkl"))
    WORDS_PATH = resource_path(os.path.join("ASSETS", "LEARNING_GAME", "words.json"))

    # --- INITIALIZATION ---
    pygame.init()
    pygame.mixer.init()
    WIDTH, HEIGHT = 1200, 700
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("HAND MASTER")

    # --- COLORS ---
    BG_DARK = (10, 10, 15)
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    NEON_GREEN = (57, 255, 20)
    GOLD = (255, 215, 0)
    NEON_BLUE = (240, 240, 240)
    PANEL_COLOR = (20, 20, 30, 180)

    # --- FONTS ---
    font_huge = pygame.font.SysFont("VanillaWhale-Regular", 120)
    font_large = pygame.font.SysFont("arialblack", 38)
    font_small = pygame.font.SysFont("consolas", 25)
    font_analyzing = pygame.font.SysFont("arialblack", 120)

    # --- LOAD SOUND EFFECTS ---
    def load_sound(filename):
        try:
            # We look for sounds specifically inside ASSETS/LEARNING_GAME/
            full_path = resource_path(os.path.join("ASSETS", "LEARNING_GAME", filename))
            if os.path.exists(full_path):
                return pygame.mixer.Sound(full_path)
            return None
        except:
            return None
        
    snd_success = load_sound("success.wav")
    snd_reset = load_sound("reset.wav")
    snd_word = load_sound("word_complete.wav")

    # --- WORD LOADER ---
    def load_sharma_words(json_path=WORDS_PATH):
        try:
            if not os.path.exists(json_path): return ["HELLO", "SIGN", "LEARN"]
            with open(json_path, 'r') as f:
                data = json.load(f)
            raw_list = list(data.keys()) if isinstance(data, dict) else data
            playable = [str(w).upper() for w in raw_list if 3 <= len(str(w)) <= 6 and str(w).isalpha()]
            return playable if playable else ["SHARMA", "PRAGYAAN"]
        except:
            return ["BACKUP", "SHARMA", "CODE"]

    # --- AI MODEL LOADING ---
    model = None
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
        else:
            print(f"❌ Critical Error: {MODEL_PATH} not found.")
    except Exception as e:
        print(f"❌ Model Error: {e}")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

    # --- UI DRAWING HELPERS ---
    def draw_glass_rect(surf, rect, color, border_color, glow=True):
        overlay = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
        pygame.draw.rect(overlay, color, (0, 0, rect[2], rect[3]), border_radius=15)
        surf.blit(overlay, (rect[0], rect[1]))
        if glow:
            pygame.draw.rect(surf, border_color, rect, 2, border_radius=15)

    def draw_grid():
        for x in range(0, WIDTH, 40):
            pygame.draw.line(screen, (20, 20, 40), (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, 40):
            pygame.draw.line(screen, (20, 20, 40), (0, y), (WIDTH, y))

    def draw_neon_hand(img_surface, hand_landmarks):
        connections = mp.solutions.hands.HAND_CONNECTIONS
        w, h = img_surface.get_width(), img_surface.get_height()
        for conn in connections:
            p1 = hand_landmarks.landmark[conn[0]]
            p2 = hand_landmarks.landmark[conn[1]]
            start = (int(p1.x * w), int(p1.y * h))
            end = (int(p2.x * w), int(p2.y * h))
            pygame.draw.line(img_surface, (0, 150, 255), start, end, 5) 
            pygame.draw.line(img_surface, CYAN, start, end, 2) 

    def draw_rounded_camera(surface, cam_surface, x, y, w, h, radius=25, border_color=(255, 0, 255)):
        cam_surface = pygame.transform.smoothscale(cam_surface, (w, h))
        mask = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, w, h), border_radius=radius)
        cam_surface.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        surface.blit(cam_surface, (x, y))
        pygame.draw.rect(surface, border_color, (x, y, w, h), 3, border_radius=radius)

    def normalize_landmarks(landmark_list):
        coords = np.array(landmark_list).reshape(-1, 3)
        wrist = coords[0]
        coords = coords - wrist
        scale = np.linalg.norm(coords[9])
        if scale == 0: scale = 1
        return (coords / scale).flatten()

    def is_open_palm(hand_landmarks):
        tips = [4, 8, 12, 16, 20]
        joints = [6, 10, 14, 18]
        extended = sum(1 for t, j in zip(tips, joints) if hand_landmarks.landmark[t].y < hand_landmarks.landmark[j].y)
        return extended >= 4

    class GameState:
        def __init__(self, pool):
            self.mode = "MENU" 
            self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            self.word_pool = pool
            self.score, self.streak, self.last_success_time = 0, 0, 0
            self.feedback = "READY"
            self.current_idx = 0
            self.current_word = ""
            self.word_progress = 0
            self.target_letter = "A"

        def pick_new_word(self):
            word = random.choice(self.word_pool)
            return word.upper()

    game = GameState(load_sharma_words())
    cap = cv2.VideoCapture(0)
    
    # Load Reference Images safely
    ref_images = {}
    for l in game.alphabet:
        img_path = os.path.join(ASSETS_DIR, f"{l}.png")
        if os.path.exists(img_path):
            try:
                ref_images[l] = pygame.transform.scale(pygame.image.load(img_path), (350, 350))
            except: pass

    running = True
    clock = pygame.time.Clock()

    while running:
        screen.fill(BG_DARK)
        draw_grid()
            
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1) 
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_e: 
                    game.mode, game.current_idx = "EASY", 0
                    game.target_letter = game.alphabet[0]
                if event.key == pygame.K_h: 
                    game.mode, game.current_idx = "HARD", 0
                    game.target_letter = game.alphabet[0]
                if event.key == pygame.K_c: 
                    game.mode, game.current_word, game.word_progress = "COMPLEX", game.pick_new_word(), 0
                    game.target_letter = game.current_word[0]
                if event.key == pygame.K_ESCAPE: 
                    game.mode = "MENU"

        if game.mode == "MENU":
            draw_glass_rect(screen, (WIDTH//2-400, HEIGHT//2-150, 800, 300), PANEL_COLOR, CYAN)
            title = font_huge.render("Hand Master", True, CYAN)
            screen.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT//2 - 120))
            hint = font_small.render("[E] EASY  |  [H] HARD  |  [C] COMPLEX", True, NEON_BLUE)
            screen.blit(hint, (WIDTH//2 - hint.get_width()//2, HEIGHT//2 + 50))
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
                
            draw_glass_rect(screen, (30, 30, WIDTH//2-60, HEIGHT-60), PANEL_COLOR, CYAN)
            draw_glass_rect(screen, (WIDTH//2+10, 30, WIDTH//2-40, HEIGHT-60), (0,0,0,100), MAGENTA)

            # UI Header Logic
            if game.mode == "COMPLEX":
                x_pos = 80
                for i, char in enumerate(game.current_word):
                    color = NEON_GREEN if i < game.word_progress else (GOLD if i == game.word_progress else (100,100,100))
                    screen.blit(font_large.render(char, True, color), (x_pos, 250))
                    x_pos += 70
                header = font_large.render(f"WORD: {game.current_word}", True, CYAN)
            else:
                header = font_large.render(f"TARGET: {game.target_letter}", True, CYAN)
                if game.mode == "EASY" and game.target_letter in ref_images:
                    screen.blit(ref_images[game.target_letter], (WIDTH//4 - 175, 180))
            
            screen.blit(header, (60, 60))
                
            # Camera Viewport logic
            CAM_W, CAM_H = 500, 375
            cam_x, cam_y = WIDTH//2 + 50, HEIGHT//2 - 180
            cam_frame = cv2.resize(frame, (CAM_W, CAM_H))
            cam_surface = pygame.surfarray.make_surface(cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB).swapaxes(0, 1))
                
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    if game.mode != "HARD":
                        draw_neon_hand(cam_surface, hand_lms)
                        
                    if is_open_palm(hand_lms):
                        if game.streak > 0 and snd_reset: snd_reset.play()
                        game.streak, game.feedback = 0, "STREAK RESET"
                        continue

                    if model is not None:
                        live_raw = [c for lm in hand_lms.landmark for c in (lm.x, lm.y, lm.z)]
                        try:
                            # Use normalize function to match model expectations
                            prediction = model.predict([normalize_landmarks(live_raw)])[0]
                            if prediction == game.target_letter:
                                game.feedback = "MATCH!"
                                if time.time() - game.last_success_time > 1.2:
                                    game.score += 10
                                    if game.mode == "COMPLEX":
                                        game.word_progress += 1
                                        if game.word_progress >= len(game.current_word):
                                            if snd_word: snd_word.play()
                                            game.current_word, game.word_progress, game.streak = game.pick_new_word(), 0, game.streak + 1
                                        else:
                                            if snd_success: snd_success.play()
                                        game.target_letter = game.current_word[game.word_progress]
                                    else:
                                        if snd_success: snd_success.play()
                                        game.current_idx = (game.current_idx + 1) % 26
                                        game.target_letter = game.alphabet[game.current_idx]
                                        game.streak += 1
                                    game.last_success_time = time.time()
                            else: 
                                game.feedback = f"Analyzing: {prediction}"
                        except Exception: pass

            draw_rounded_camera(screen, cam_surface, cam_x, cam_y, CAM_W, CAM_H, radius=20, border_color=CYAN)

            # Stats Display
            score_txt = font_small.render(f"SCORE: {game.score}", True, NEON_GREEN)
            streak_txt = font_small.render(f"STREAK: {game.streak}", True, CYAN)
            screen.blit(score_txt, (WIDTH - 200, 50))
            screen.blit(streak_txt, (WIDTH - 200, 80))

            feedback_surf = font_large.render(game.feedback, True, NEON_BLUE)
            screen.blit(feedback_surf, (WIDTH//2 + 100, 560))
                
        pygame.display.flip()
        clock.tick(30)
        
    cap.release()
    pygame.display.quit()

if __name__ == "__main__":
    run_hand_master()