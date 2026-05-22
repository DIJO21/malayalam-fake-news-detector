import os
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from config import Config
from preprocessing.text_cleaner import TextCleaner
from utils.visualization import plot_confusion_matrix, plot_training_history

# Enable TF32 for faster training on Ampere GPUs (if available)
torch.backends.cuda.matmul.allow_tf32 = True

def load_data():
    cleaner = TextCleaner()
    print("Loading data...")
    if os.path.exists(Config.PROCESSED_DATA_PATH):
        df = pd.read_csv(Config.PROCESSED_DATA_PATH)
    else:
        df = pd.read_csv(Config.RAW_DATA_PATH)
        df['cleaned_text'] = df['headline'].apply(cleaner.clean_malayalam_text)
        os.makedirs(os.path.dirname(Config.PROCESSED_DATA_PATH), exist_ok=True)
        df.to_csv(Config.PROCESSED_DATA_PATH, index=False)
        
    # Assume target column is 'label' and text is 'headline' or 'text'
    text_col = 'cleaned_text' if 'cleaned_text' in df.columns else 'headline'
    
    # Filter empty texts
    df = df[df[text_col].notna()]
    df = df[df[text_col].str.strip() != ""]
    
    return df[text_col].values, df['label'].values

def train_transformer():
    print(f"\n{'='*50}\nTraining {Config.MBERT_MODEL_NAME} using PyTorch...\n{'='*50}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    X, y = load_data()
    
    # 1. Compute Class Weights (to fix majority-class collapse)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Computed Class Weights: {class_weights}")
    
    # 2. Tokenize using MuRIL
    print("Loading tokenizer...")
    tokenizer_dir = os.path.join(Config.SAVED_MODELS_DIR, 'mbert_tokenizer')
    if not os.path.exists(tokenizer_dir):
        tokenizer_dir = os.path.join(Config.SAVED_MODELS_DIR, 'mbert_model')
        
    if os.path.exists(tokenizer_dir) and os.path.exists(os.path.join(tokenizer_dir, 'tokenizer_config.json')):
        print(f"Loading tokenizer from local path: {tokenizer_dir}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    else:
        print(f"Loading tokenizer from pretrained: {Config.MBERT_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(Config.MBERT_MODEL_NAME)
    
    encoded = tokenizer(
        list(X),
        padding='max_length',
        truncation=True,
        max_length=Config.MAX_SEQUENCE_LENGTH,
        return_tensors='pt'
    )
    
    dataset = TensorDataset(encoded['input_ids'], encoded['attention_mask'], torch.tensor(y))
    
    # 3. Stratified Train/Val Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    
    # 4. Load MuRIL Model
    model_dir = os.path.join(Config.SAVED_MODELS_DIR, 'mbert_model')
    if os.path.exists(model_dir) and os.path.exists(os.path.join(model_dir, 'config.json')) and (os.path.exists(os.path.join(model_dir, 'model.safetensors')) or os.path.exists(os.path.join(model_dir, 'pytorch_model.bin'))):
        # To avoid Windows memory-mapped file locking (os error 1224) when saving to the same folder:
        # copy checkpoint files to a temp load folder and load from there.
        import shutil
        temp_load_dir = os.path.join(Config.SAVED_MODELS_DIR, 'mbert_model_load_temp')
        if os.path.exists(temp_load_dir):
            try:
                shutil.rmtree(temp_load_dir)
            except Exception:
                pass
        os.makedirs(temp_load_dir, exist_ok=True)
        for f in os.listdir(model_dir):
            shutil.copy2(os.path.join(model_dir, f), os.path.join(temp_load_dir, f))
            
        print(f"Resuming training: Copied checkpoint to temp path and loading from {temp_load_dir}")
        model = AutoModelForSequenceClassification.from_pretrained(
            temp_load_dir,
            num_labels=Config.CLASSES
        )
    else:
        print(f"Training from scratch using pretrained: {Config.MBERT_MODEL_NAME}")
        model = AutoModelForSequenceClassification.from_pretrained(
            Config.MBERT_MODEL_NAME, 
            num_labels=Config.CLASSES
        )
    model.to(device)
    
    # 5. Advanced Optimization Setup
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01)
    
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.1), 
        num_training_steps=total_steps
    )
    
    # Custom Loss with Class Weights
    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # AMP (Mixed Precision) for faster training
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3 # Early stopping
    
    # 6. Training Loop
    for epoch in range(Config.EPOCHS):
        print(f"\n======== Epoch {epoch + 1} / {Config.EPOCHS} ========")
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for step, batch in enumerate(train_loader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            
            optimizer.zero_grad()
            
            # Use Mixed Precision if GPU is available
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(b_input_ids, attention_mask=b_input_mask)
                    loss = loss_fct(outputs.logits, b_labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(b_input_ids, attention_mask=b_input_mask)
                loss = loss_fct(outputs.logits, b_labels)
                loss.backward()
                optimizer.step()
                
            scheduler.step()
            total_loss += loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(b_labels.cpu().numpy())
            
            if (step + 1) % 5 == 0:
                print(f"    Step {step + 1} / {len(train_loader)} - Loss: {loss.item():.4f}")
                
        avg_train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        
        print(f"  Average training loss: {avg_train_loss:.4f}")
        print(f"  Average training accuracy: {train_acc:.4f}")
        
        # Validation Phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                
                outputs = model(b_input_ids, attention_mask=b_input_mask)
                loss = loss_fct(outputs.logits, b_labels)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(b_labels.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        print(f"  Validation Accuracy: {val_acc:.4f}")
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            model_dir = os.path.join(Config.SAVED_MODELS_DIR, 'mbert_model')
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir) # Save tokenizer with model!
            print("  [SUCCESS] Model checkpoint saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("  [!] Early stopping triggered.")
                break

    # 7. Final Evaluation on Best Model
    print("\n--- MuRIL Performance ---")
    precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary')
    print(f"Accuracy:  {val_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    cm = confusion_matrix(val_labels, val_preds)
    print("Confusion Matrix:")
    print(cm)
    
    plot_confusion_matrix(cm, "MuRIL")
    plot_training_history(history, "MuRIL")

if __name__ == "__main__":
    Config.setup()
    
    # We only train the optimized MuRIL Transformer pipeline. 
    # The older Keras RNNs have been permanently retired due to majority-class collapse.
    train_transformer()
    print("\nTraining completed. Models and metrics saved.")
