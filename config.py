import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'malayalam_dataset_fast.csv')
    PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_data.csv')
    SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    
    # Text Processing
    MAX_SEQUENCE_LENGTH = 64
    MAX_WORDS = 20000  # For GRU / BiLSTM embedding
    EMBEDDING_DIM = 100
    
    # Model Hyperparameters
    BATCH_SIZE = 8
    EPOCHS = 10
    LEARNING_RATE = 2e-5  # For mBERT
    LEARNING_RATE_DL = 1e-3 # For BiLSTM/GRU
    
    # Classes
    CLASSES = 2
    
    # Transformers (Upgraded to MuRIL for Indian Languages)
    MBERT_MODEL_NAME = "google/muril-base-cased"
    
    # RAG Configuration
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    @staticmethod
    def setup():
        os.makedirs(Config.SAVED_MODELS_DIR, exist_ok=True)
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)

