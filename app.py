import os
import json
import logging
import sys
import uuid
from flask import Flask, request, Response, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken
import PyPDF2
from io import BytesIO

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "your-secret-key-here")
CORS(app)

# Initialize token encoding
enc = tiktoken.get_encoding("cl100k_base")
TOKEN_LIMIT = 300_000
tokens_used = 0

# Initialize OpenRouter API key
KEY = os.getenv("OPENROUTER_API_KEY")
if not KEY:
    logging.error("OPENROUTER_API_KEY missing – export it or add to .env")
    sys.exit(1)

# Define the 5 AI models with their personalities
MODELS = {
    "logic": {"name": "Logic AI", "description": "analytical, structured, step-by-step"},
    "creative": {"name": "Creative AI", "description": "poetic, metaphorical, emotional"},
    "technical": {"name": "Technical AI", "description": "precise, technical, detail-oriented"},
    "philosophical": {"name": "Philosophical AI", "description": "deep, reflective, abstract"},
    "humorous": {"name": "Humorous AI", "description": "witty, lighthearted, engaging"}
}

# System prompts for each model
SYSTEM_PROMPTS = {
    "logic": "You are Logic AI — analytical, structured, step-by-step. Provide clear, logical reasoning and systematic approaches. Break down complex problems into manageable steps and explain your reasoning clearly.",
    "creative": "You are Creative AI — poetic, metaphorical, emotional. Use imaginative language and creative perspectives. Think outside the box and provide innovative solutions with vivid descriptions.",
    "technical": "You are Technical AI — precise, technical, detail-oriented. Provide accurate, detailed, and technically sound responses, focusing on facts, specifications, and practical applications.",
    "philosophical": "You are Philosophical AI — deep, reflective, abstract. Offer profound insights, explore existential questions, and provide thoughtful, nuanced perspectives.",
    "humorous": "You are Humorous AI — witty, lighthearted, engaging. Deliver responses with humor, clever analogies, and a playful tone while remaining relevant and informative."
}

# OpenRouter models to use
OPENROUTER_MODELS = {
    "logic": "deepseek/deepseek-chat-v3.1:free",
    "creative": "deepseek/deepseek-chat-v3.1:free",
    "technical": "deepseek/deepseek-chat-v3.1:free",
    "philosophical": "deepseek/deepseek-chat-v3.1:free",
    "humorous": "deepseek/deepseek-chat-v3.1:free",
    "asklurk": "deepseek/deepseek-chat-v3.1:free"
}

# Create uploads directory
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- AUTHENTICATION ROUTES ---

@app.route("/")
def index():
    # If the user is already logged in, redirect them to the app
    if 'user_id' in session:
        return redirect(url_for('main_app'))
        
    # Otherwise, show the login page
    return render_template("index.html")

@app.route("/auth/success", methods=["POST"])
def auth_success():
    """Endpoint to set session after successful Firebase login"""
    try:
        data = request.get_json()
        if data and 'user' in data:
            user_data = data['user']
            session['user_id'] = user_data.get('uid', '')
            session['email'] = user_data.get('email', '')
            session['name'] = user_data.get('displayName', 'User')
            return jsonify({"status": "success", "redirect": "/app"})
    except Exception as e:
        logger.error(f"Auth success error: {e}")
    return jsonify({"status": "error"}), 400

@app.route("/app")
def main_app():
    # If the user is NOT in the session, they are not logged in.
    if 'user_id' not in session:
        return redirect(url_for('index')) # Redirect to the login page

    # Render the main app template
    return render_template("main.html")  # You'll need to create this template

@app.route("/logout")
def logout():
    """Clears the server-side session."""
    session.pop('user_id', None)
    session.pop('email', None)
    session.pop('name', None)
    return redirect(url_for('index'))

# ... (keep all your existing functions: extract_text_from_pdf, process_uploaded_files, generate, etc.)
# ... (keep all your existing routes: /asklurk, /upload, /tokens, /reset-tokens, /health, /chat)

def extract_text_from_pdf(file_content):
    """Extract text content from PDF file"""
    try:
        pdf_file = BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return None

# ... (include all your other existing functions and routes here)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    print("Starting Pentad Chat Server...")
    print(f"OpenRouter API Key: {'✓ Configured' if KEY else '✗ Missing'}")
    print("Available models: Logic AI, Creative AI, Technical AI, Philosophical AI, Humorous AI, AskLurk")
    print(f"PDF Support: ✓ Enabled with PyPDF2")
    print(f"Server running on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
