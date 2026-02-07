import random

# ---------------- INTENTS DATA ----------------
intents = {
    "intents": [

        # ---------- GREETING ----------
        {
            "tag": "greeting",
            "patterns": [
                "hi", "hey", "hello", "good morning", "good evening",
                "hey there", "what's up", "can i ask you something",
                "i have a doubt", "yo", "greetings"
            ],
            "responses": [
                "Hello! How can I assist you today?",
                "Hi there! What can I do for you?",
                "Greetings! How may I help you?"
            ]
        },

        # ---------- GOODBYE ----------
        {
            "tag": "goodbye",
            "patterns": [
                "bye", "see you later", "goodbye", "take care",
                "catch you later", "talk to you soon", "farewell"
            ],
            "responses": [
                "Goodbye! Have a great day!",
                "See you later! Take care!",
                "Farewell! Let me know if you need anything else."
            ]
        },

        # ---------- THANKS ----------
        {
            "tag": "thanks",
            "patterns": [
                "thanks", "thank you", "thanks a lot", "much appreciated",
                "thank you very much", "that's helpful", "thanks for the help",
                "thx", "cheers"
            ],
            "responses": [
                "You're welcome!",
                "No problem!",
                "Anytime!",
                "Glad I could help!",
                "Happy to assist!"
            ]
        },

        # ---------- INTRODUCTION ----------
        {
            "tag": "introduction",
            "patterns": [
                "who are you", "what is your name", "introduce yourself",
                "tell me about yourself", "what do you do",
                "who am i talking to", "your introduction"
            ],
            "responses": [
                "Hey there! I am Pragyaan, an AI-based platform focused on accessibility and smart interaction. We provide features like hand gesture recognition, virtual mouse, and more."
            ]
        },

        # ---------- SMALL TALK ----------
        {
            "tag": "small_talk",
            "patterns": [
                "how are you", "are you a bot", "are you real",
                "do you exist", "are you human"
            ],
            "responses": [
                "Hey, I'm doing great. Ready to help you explore Pragyaan!",
                "I am an AI chatbot built to assist you on this website."
            ]
        },

        # ---------- ABOUT PRAGYAAN ----------
        {
            "tag": "about_pragyaan",
            "patterns": [
                "what is pragyaan", "tell me about pragyaan",
                "what does pragyaan do", "information about pragyaan",
                "what is this website about", "explain pragyaan",
                "details about pragyaan", "purpose of pragyaan"
            ],
            "responses": [
                "Pragyaan is a smart AI-driven platform designed to help users interact using voice, text, and gestures. Our aim is to improve accessibility and learning using AI."
            ]
        },

        # ---------- TEAM ----------
        {
            "tag": "team_info",
            "patterns": [
                "who developed this website", "provide team details",
                "who created this", "who are the creators",
                "who made pragyaan", "tell me about the developers"
            ],
            "responses": [
                "This website is developed by Team Pragyaan, consisting of passionate individuals dedicated to enhancing accessibility through AI technology."
            ]
        },

        # ---------- HOW TO USE ----------
        {
            "tag": "how_to_use",
            "patterns": [
                "how to use pragyaan", "guide me on using this website",
                "instructions for using pragyaan", "how do i navigate",
                "help me understand pragyaan", "how does this work"
            ],
            "responses": [
                "To use Pragyaan, explore the features available on the website. You can interact using voice commands, text input, or hand gestures. Each feature has its own instructions."
            ]
        },

        # ---------- PAYMENTS ----------
        {
            "tag": "payments",
            "patterns": [
                "is this website free", "are there charges",
                "do i need to pay", "what is the cost",
                "are there subscription fees", "is pragyaan paid"
            ],
            "responses": [
                "Pragyaan is completely free to use! There are no charges or subscription fees."
            ]
        },

        # ---------- FEATURES ----------
        {
            "tag": "features",
            "patterns": [
                "what features does pragyaan offer", "list the features",
                "tell me about functionalities", "what can i do here",
                "explain the features", "capabilities of pragyaan"
            ],
            "responses": [
                "Pragyaan offers gesture recognition, virtual mouse control, voice commands, text input, and a game to learn sign language."
            ]
        },

        # ---------- NAVIGATION ----------
        {
            "tag": "navigation_help",
            "patterns": [
                "where is chatbot", "where can i play games",
                "where is gesture recognition", "how to access virtual mouse",
                "help me navigate", "show me where features are"
            ],
            "responses": [
                "You can find the chatbot at the bottom right. Games are under the 'Games' tab, gesture recognition is on the 'Gesture Recognition' page, and the virtual mouse is in the 'Virtual Mouse' section."
            ]
        },

        # ---------- CHATBOT ----------
        {
            "tag": "chatbot_feature",
            "patterns": [
                "how does the chatbot work", "explain chatbot feature",
                "what can i do with chatbot", "tell me about chatbot",
                "how to interact with chatbot", "chatbot functionality"
            ],
            "responses": [
                "The chatbot allows you to interact with Pragyaan using text input. You can ask questions, seek assistance, and get information about the website."
            ]
        },

        # ---------- GESTURE ----------
        {
            "tag": "gesture_recognition",
            "patterns": [
                "how does gesture recognition work", "explain gesture recognition",
                "what can i do with gestures", "tell me about gesture recognition",
                "how to use gesture recognition", "what gestures are recognized"
            ],
            "responses": [
                "Gesture recognition lets you interact with Pragyaan using hand gestures via webcam. The system detects movements to perform actions, enhancing accessibility."
            ]
        },

        # ---------- HELP ----------
        {
            "tag": "help",
            "patterns": [
                "i need help", "can you assist me",
                "help me with this website", "i am confused",
                "i need guidance", "i need support"
            ],
            "responses": [
                "Sure! I'm here to help. Ask me any questions about Pragyaan, its features, or navigation."
            ]
        },

        # ---------- FEEDBACK ----------
        {
            "tag": "feedback",
            "patterns": [
                "i want to give feedback", "how can i provide feedback",
                "where can i leave comments", "i have suggestions",
                "can i share my thoughts", "feedback options"
            ],
            "responses": [
                "We appreciate your feedback! You can share comments and suggestions through the 'Contact Us' section."
            ]
        },

        # ---------- VIRTUAL MOUSE ----------
        {
            "tag": "virtual_mouse",
            "patterns": [
                "virtual mouse", "about virtual mouse",
                "how to use virtual mouse", "explain virtual mouse feature",
                "what can i do with virtual mouse", "how does virtual mouse work"
            ],
            "responses": [
                "The virtual mouse allows cursor control using hand gestures. Move with index finger, left click by pinching index and thumb, right click with middle finger and thumb, scroll using thumb up/down, and drag with index and thumb."
            ]
        },

        # ---------- TROUBLESHOOT ----------
        {
            "tag": "troubleshooting",
            "patterns": [
                "website is not working", "i am facing issues",
                "something is wrong", "feature is not responding",
                "technical support", "website problem"
            ],
            "responses": [
                "Sorry you're experiencing issues. Try refreshing or clearing your cache. If it persists, contact support via 'Contact Us'."
            ]
        },

        # ---------- GAME ----------
        {
            "tag": "hand_master",
            "patterns": [
                "handmaster", "hand master", "game",
                "sign language game", "what is hand master"
            ],
            "responses": [
                "Hand Master is a game designed to help users learn sign language. It has three levels: easy, medium, and hard."
            ]
        },

        # ---------- SOS ----------
        {
            "tag": "emergency_sos",
            "patterns": [
                "sos", "what is sos", "how to use sos",
                "features of sos", "what does sos do"
            ],
            "responses": [
                "The SOS system automatically sends an emergency SMS when a closed fist is detected for 5 seconds, along with the user's location."
            ]
        },

        # ---------- FALLBACK ----------
        {
            "tag": "fallback",
            "patterns": [],
            "responses": [
                "I'm sorry, I didn't understand that. Could you rephrase?",
                "I'm not sure I follow. Can you ask differently?",
                "Apologies, I don't have an answer for that.",
                "I'm here to help with Pragyaan-related questions."
            ]
        }
    ]
}

# ---------------- CORE FUNCTION ----------------
def get_response(user_input: str) -> str:
    user_input = user_input.lower().strip()

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            if pattern in user_input:
                return random.choice(intent["responses"])

    # fallback
    for intent in intents["intents"]:
        if intent["tag"] == "fallback":
            return random.choice(intent["responses"])

    return "Sorry, I didn't understand that."