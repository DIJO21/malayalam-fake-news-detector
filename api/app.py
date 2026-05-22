from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import sys

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from preprocessing.text_cleaner import TextCleaner

app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')
CORS(app)

# Load MuRIL Model
print("Loading model and tokenizer...")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.abspath(os.path.join(Config.SAVED_MODELS_DIR, 'mbert_model'))
    print(f"Loading model and tokenizer from {model_path}")
    if os.path.exists(os.path.join(model_path, 'tokenizer_config.json')):
        print("Loading tokenizer from local model path...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        print(f"Loading tokenizer from pretrained: {Config.MBERT_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(Config.MBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Warning: Model could not be loaded. Please train the models first. Error: {e}")
    tokenizer = None
    model = None

cleaner = TextCleaner()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "running", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or tokenizer is None:
        return jsonify({"error": "Model not loaded. Please train first."}), 500
        
    data = request.get_json(force=True)
    text = data.get('text', '')
    
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400
        
    # Preprocess
    cleaned_text = cleaner.clean_malayalam_text(text)
    
    if not cleaned_text.strip():
         return jsonify({"error": "Text is empty after cleaning"}), 400
         
    # Tokenize
    encoded = tokenizer(
        [cleaned_text],
        padding='max_length',
        truncation=True,
        max_length=Config.MAX_SEQUENCE_LENGTH,
        return_tensors='pt'
    )
    
    # Predict
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
    
    pred_class = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
    confidence = float(probs[pred_class])
    
    result = "real" if pred_class == 1 else "fake"
    
    # RAG Integration: Live Fact Checking
    evidence_list = []
    reasoning = ""
    
    try:
        from retrieval.rag_engine import RAGEngine
        rag = RAGEngine()
        
        # 1. Fetch live news
        live_articles = rag.fetch_live_news(cleaned_text)
        
        # 2. Compute similarity
        evidence_list = rag.compute_similarity(cleaned_text, live_articles)
        
        # 3. LLM Reasoning
        if evidence_list:
            reasoning = rag.verify_facts_with_llm(cleaned_text, evidence_list)
        else:
            reasoning = "No relevant real-time evidence found on global news networks to corroborate this text."
            
    except Exception as e:
        print(f"RAG Engine Error: {e}")
        reasoning = "Live Fact-Checking service is currently unavailable."
    
    return jsonify({
        "result": result,
        "confidence": round(confidence, 4),
        "evidence": evidence_list[:3],  # Return top 3
        "reasoning": reasoning
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
