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

# --- ADDED: FIREBASE ADMIN IMPORTS ---
import firebase_admin
from firebase_admin import credentials, auth
# -------------------------------------

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
# IMPORTANT: The secret key is CRUCIAL for Flask sessions (authentication)
app.secret_key = os.environ.get("SESSION_SECRET", "a_strong_and_unpredictable_secret_key_12345")
CORS(app)

# --- ADDED: FIREBASE ADMIN INITIALIZATION ---
# 🚨 SECURITY CRITICAL: Replace the path below with the actual location of your Firebase service account JSON file.
try:
    # Assuming serviceAccountKey.json is in the same directory as app.py
    cred = credentials.Certificate("serviceAccountKey.json") 
    # Use a unique name for the app instance
    firebase_admin.initialize_app(cred, name="auth_backend") 
    auth_app = firebase_admin.get_app("auth_backend")
    logger.info("Firebase Admin SDK initialized successfully.")
except Exception as e:
    logger.error(f"FATAL ERROR: Failed to initialize Firebase Admin SDK. Check serviceAccountKey.json path. Error: {e}")
# --------------------------------------------

# Initialize token encoding
enc = tiktoken.get_encoding("cl100k_base")
TOKEN_LIMIT = 300_000
tokens_used = 0

# Initialize OpenRouter API key
KEY = os.getenv("OPENROUTER_API_KEY")
if not KEY:
    logging.error("OPENROUTER_API_KEY missing – export it or add to .env")

# Define the 5 AI models with their personalities (MODELS, SYSTEM_PROMPTS, OPENROUTER_MODELS kept here)
MODELS = {
    "logic": {"name": "Logic AI", "description": "analytical, structured, step-by-step"},
    "creative": {"name": "Creative AI", "description": "poetic, metaphorical, emotional"},
    "technical": {"name": "Technical AI", "description": "precise, technical, detail-oriented"},
    "philosophical": {"name": "Philosophical AI", "description": "deep, reflective, abstract"},
    "humorous": {"name": "Humorous AI", "description": "witty, lighthearted, engaging"}
}

SYSTEM_PROMPTS = {
    "logic": "You are Logic AI — analytical, structured, step-by-step. Provide clear, logical reasoning and systematic approaches. Break down complex problems into manageable steps and explain your reasoning clearly.",
    "creative": "You are Creative AI — poetic, metaphorical, emotional. Use imaginative language and creative perspectives. Think outside the box and provide innovative solutions with vivid descriptions.",
    "technical": "You are Technical AI — precise, technical, detail-oriented. Provide accurate, detailed, and technically sound responses, focusing on facts, specifications, and practical applications.",
    "philosophical": "You are Philosophical AI — deep, reflective, abstract. Offer profound insights, explore existential questions, and provide thoughtful, nuanced perspectives.",
    "humorous": "You are Humorous AI — witty, lighthearted, engaging. Deliver responses with humor, clever analogies, and a playful tone while remaining relevant and informative."
}

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
    """Renders the login page or redirects to the app if already logged in."""
    # This renders templates/index.html
    if 'user_id' in session:
        return redirect(url_for('main_app'))
        
    return render_template("index.html")

# 🚨 NEW: SECURE ENDPOINT to verify Firebase token and set Flask session
@app.route("/auth/google_verify", methods=["POST"]) 
def google_verify():
    """Verifies the Firebase ID Token sent by the client and sets the server session."""
    try:
        data = request.get_json()
        id_token = data.get('idToken') # Get the secure ID token

        if not id_token:
            return jsonify({"status": "error", "message": "No ID token provided"}), 400

        # 1. CRITICAL: VERIFY the ID Token with Google/Firebase
        decoded_token = auth.verify_id_token(id_token, app=auth_app)
        
        # 2. Set the secure Flask session
        session['user_id'] = decoded_token['uid']
        session['email'] = decoded_token.get('email', '')
        session['name'] = decoded_token.get('name', 'User')

        logger.info(f"User verified and session set: {session['user_id']}")
        
        # 3. Success response with the redirect URL
        return jsonify({"status": "success", "redirect": "/app"})
        
    except firebase_admin.exceptions.FirebaseError as e:
        logger.error(f"Firebase verification error: {e}")
        return jsonify({"status": "error", "message": "Token verification failed. Client must sign out and retry."}), 401
    except Exception as e:
        logger.error(f"General auth error: {e}")
        return jsonify({"status": "error", "message": "Server error during verification"}), 500

@app.route("/app")
def main_app():
    """Protected main application page (Pentad Chat UI)."""
    # Protection check: If user is not in session, redirect to login
    if 'user_id' not in session:
        return redirect(url_for('index'))

    # Render the main chat UI
    # NOTE: This requires 'main.html' to be in your 'templates' folder.
    return render_template("main.html")

@app.route("/logout")
def logout():
    """Clears the server-side session."""
    session.pop('user_id', None)
    session.pop('email', None)
    session.pop('name', None)
    # Redirect back to the login page
    return redirect(url_for('index'))

# ------------------------------------
# --- AI FUNCTIONALITY (Rest of your existing code remains unchanged) ---

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

def process_uploaded_files(files):
    # This is a stub for file processing logic not fully implemented elsewhere
    file_contents = []
    
    for file in files:
        try:
            file_content = file.read()
            filename = file.filename.lower()
            
            if filename.endswith('.pdf'):
                text_content = extract_text_from_pdf(file_content)
                if text_content:
                    file_contents.append(f"PDF Content from '{file.filename}':\n{text_content}\n")
                else:
                    file_contents.append(f"PDF file '{file.filename}' (could not extract text)")
            
            elif filename.endswith(('.txt', '.doc', '.docx')):
                try:
                    if filename.endswith('.txt'):
                        text_content = file_content.decode('utf-8')
                        file_contents.append(f"Text Content from '{file.filename}':\n{text_content}\n")
                    else:
                        file_contents.append(f"Document file '{file.filename}' (content processing not implemented)")
                except:
                    file_contents.append(f"Document file '{file.filename}' (could not read content)")
            
            elif filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                file_contents.append(f"Image file '{file.filename}'")
            
            else:
                file_contents.append(f"File '{file.filename}'")
                
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            file_contents.append(f"File '{file.filename}' (error processing)")
    
    return file_contents


def generate(bot_name: str, system: str, user: str, file_contents: list = None):
    """Generate AI response for a specific bot using OpenRouter"""
    global tokens_used
    client = None
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=KEY,
            timeout=60.0 
        )
        
        # Add file contents to the user prompt if available
        full_user_prompt = user
        if file_contents:
            file_context = "\n\n".join(file_contents)
            full_user_prompt = f"{user}\n\nAttached files content:\n{file_context}"
        
        # Calculate tokens for the request
        system_tokens = len(enc.encode(system))
        user_tokens = len(enc.encode(full_user_prompt))
        tokens_used += system_tokens + user_tokens
        
        model = OPENROUTER_MODELS.get(bot_name, "deepseek/deepseek-chat-v3.1:free")
        logger.info(f"Generating response for {bot_name} using model {model}")
        
        stream = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost:5000",
                "X-Title": "Pentad-Chat"
            },
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": full_user_prompt}
            ],
            temperature=0.7,
            max_tokens=1500,  
            stream=True,
        )
        
        bot_tokens = 0
        full_response = ""
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                delta = chunk.choices[0].delta.content
                full_response += delta
                bot_tokens += len(enc.encode(delta))
                yield f"data: {json.dumps({'bot': bot_name, 'text': delta})}\n\n"
            
            if chunk.choices and chunk.choices[0].finish_reason:
                break
        
        tokens_used += bot_tokens
        logger.info(f"Completed generation for {bot_name}, tokens used: {bot_tokens}")
        yield f"data: {json.dumps({'bot': bot_name, 'done': True, 'tokens': tokens_used})}\n\n"
        
    except Exception as exc:
        logger.error(f"Error generating response for {bot_name}: {str(exc)}")
        error_msg = str(exc)
        if "401" in error_msg:
            error_msg = "Authentication failed. Invalid OpenRouter API key."
        elif "429" in error_msg:
            error_msg = "Rate limit exceeded. Please try again later."
        elif "404" in error_msg:
            error_msg = "Model not found or unavailable."
        else:
            error_msg = f"Failed to generate response: {error_msg}"
            
        yield f"data: {json.dumps({'bot': bot_name, 'error': error_msg})}\n\n"
    finally:
        if client:
            try:
                client.close()
            except:
                pass

@app.route("/asklurk", methods=["POST"])
def asklurk():
    """Synthesize the best answer from all AI responses"""
    if 'user_id' not in session:
        return jsonify(best="", error="Unauthorized"), 401
        
    try:
        data = request.json or {}
        answers = data.get("answers", {})
        prompt = data.get("prompt", "")
        
        if not answers:
            return jsonify(best="", error="No responses to analyze"), 400
        
        try:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=KEY,
                timeout=30.0
            )
            
            merged_content = f"Original question: {prompt}\n\n"
            for key, response in answers.items():
                if key in MODELS:
                    merged_content += f"## {MODELS[key]['name']}:\n{response}\n\n"
            
            response = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "http://localhost:5000",
                    "X-Title": "Pentad-Chat"
                },
                model=OPENROUTER_MODELS["asklurk"],
                messages=[
                    {
                        "role": "system",
                        "content": """You are AskLurk - an expert AI synthesizer. Your task is to analyze responses from Logic AI, Creative AI, Technical AI, Philosophical AI, and Humorous AI to create the single best answer. 
                        
                        Combine the logical reasoning, creative insights, technical accuracy, philosophical depth, and humorous engagement to provide a comprehensive, well-structured response that leverages the strengths of all approaches.
                        
                        Structure your response to be insightful, engaging, and balanced."""
                    },
                    {
                        "role": "user",
                        "content": f"""Please analyze these AI responses to the question: "{prompt}"

Here are the responses:
{merged_content}

Please provide the best synthesized answer that combines the strengths of all AI responses:"""
                    }
                ],
                temperature=0.3,
                max_tokens=1500,
            )
            
            best_answer = response.choices[0].message.content
            asklurk_tokens = len(enc.encode(best_answer))
            global tokens_used
            tokens_used += asklurk_tokens
            
            return jsonify(best=best_answer, tokens_used=tokens_used)
            
        except Exception as e:
            logger.error(f"AskLurk error: {str(e)}")
            if answers:
                first_response = next(iter(answers.values()))
                return jsonify(best=f"Fallback - Using first response:\n\n{first_response}", error="AI synthesis failed")
            return jsonify(best="", error="No responses available for synthesis")
        
    except Exception as e:
        logger.error(f"AskLurk error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route("/upload", methods=["POST"])
def upload():
    """Handle file uploads"""
    if 'user_id' not in session:
        return jsonify(urls=[], error="Unauthorized"), 401
        
    try:
        if 'files' not in request.files:
            return jsonify(urls=[], error="No files provided"), 400
        
        files = request.files.getlist('files')
        urls = []
        
        for file in files:
            if file.filename == '':
                continue
            
            allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.pdf', '.txt', '.doc', '.docx'}
            ext = os.path.splitext(file.filename)[1].lower()
            
            if ext not in allowed_extensions:
                continue
                
            name = f"{uuid.uuid4().hex}{ext}"
            path = os.path.join(UPLOAD_FOLDER, name)
            
            try:
                file.save(path)
                urls.append(f"/static/uploads/{name}")
            except Exception as e:
                logger.error(f"Error saving file {file.filename}: {str(e)}")
        
        return jsonify(urls=urls)
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route("/tokens", methods=["GET"])
def get_tokens():
    """Endpoint to get current token usage"""
    return jsonify({
        "tokens_used": tokens_used,
        "token_limit": TOKEN_LIMIT,
        "remaining_tokens": TOKEN_LIMIT - tokens_used,
        "usage_percentage": (tokens_used / TOKEN_LIMIT) * 100
    })

@app.route("/reset-tokens", methods=["POST"])
def reset_tokens():
    """Endpoint to reset token counter"""
    if 'user_id' not in session:
        return jsonify(message="Unauthorized"), 401
        
    global tokens_used
    tokens_used = 0
    return jsonify({"message": "Token counter reset", "tokens_used": tokens_used})

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "api_key_configured": bool(KEY),
        "models_configured": len(OPENROUTER_MODELS),
        "tokens_used": tokens_used
    })

@app.route("/chat", methods=["POST"])
def chat():
    """Main chat endpoint for all AI models"""
    if 'user_id' not in session:
        return jsonify(error="Unauthorized. Please log in."), 401
        
    try:
        data = request.json or {}
        prompt = data.get("prompt", "").strip()
        fileUrls = data.get("fileUrls", [])
        
        if not prompt and not fileUrls:
            return jsonify(error="Empty prompt and no files provided"), 400
        
        if tokens_used >= TOKEN_LIMIT:
            return jsonify(error=f"Token limit reached ({tokens_used}/{TOKEN_LIMIT})"), 429
        
        # Process uploaded files (logic kept from previous version)
        file_contents = []
        if fileUrls:
            for file_url in fileUrls:
                try:
                    # Strip the /static/uploads/ prefix to get the file path
                    file_path = file_url.replace('/static/uploads/', '')
                    full_path = os.path.join(UPLOAD_FOLDER, file_path)
                    
                    if os.path.exists(full_path):
                        with open(full_path, 'rb') as f:
                            file_content = f.read()
                        
                        filename = file_path.lower()
                        if filename.endswith('.pdf'):
                            text_content = extract_text_from_pdf(file_content)
                            if text_content:
                                file_contents.append(f"PDF Content from '{file_path}':\n{text_content}\n")
                        elif filename.endswith('.txt'):
                            text_content = file_content.decode('utf-8')
                            file_contents.append(f"Text Content from '{file_path}':\n{text_content}\n")
                        else:
                            file_contents.append(f"File '{file_path}'")
                            
                except Exception as e:
                    logger.error(f"Error processing file {file_url}: {str(e)}")
                    file_contents.append(f"File '{file_url}' (error processing)")

        def event_stream():
            generators = {}
            for key in MODELS.keys():
                generators[key] = generate(key, SYSTEM_PROMPTS[key], prompt, file_contents)
            
            active_bots = list(MODELS.keys())
            completed_bots = set()
            
            while active_bots:
                for bot_name in active_bots[:]:
                    try:
                        chunk = next(generators[bot_name])
                        yield chunk
                        
                        try:
                            chunk_data = json.loads(chunk.split('data: ')[1])
                            if chunk_data.get('done') or chunk_data.get('error'):
                                completed_bots.add(bot_name)
                                active_bots.remove(bot_name)
                        except:
                            pass
                            
                    except StopIteration:
                        completed_bots.add(bot_name)
                        active_bots.remove(bot_name)
                        
                    except Exception as e:
                        logger.error(f"Error streaming for {bot_name}: {str(e)}")
                        completed_bots.add(bot_name)
                        active_bots.remove(bot_name)
            
            yield f"data: {json.dumps({'all_done': True, 'tokens': tokens_used})}\n\n"

        return Response(
            event_stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            },
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    print("Starting Pentad Chat Server...")
    print(f"Server running on http://0.0.0.0:{port}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
