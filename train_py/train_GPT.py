import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import cohen_kappa_score, accuracy_score
from tqdm import tqdm


class EssayDataset(Dataset):
    def __init__(self, encodings, scores=None):
        self.encodings = encodings
        self.scores = scores

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.scores is not None:
            item['labels'] = torch.tensor(self.scores[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


# Model initialization
def model_init():
    model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=1)
    model.config.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))

    # Apply dropout variations
    model.config.hidden_dropout_prob = 0.5
    model.config.attention_probs_dropout_prob = 0.6

    # Label smoothing
    model.config.label_smoothing = 0.1

    model.config.noise_sigma = 0.02  # Add Gaussian noise with standard deviation of 0.02 to embeddings

    # Stochastic depth
    model.config.stochastic_depth_prob = 0.1  # 10% chance of dropping a layer

    return model

def calculate_qwk(y_true, y_pred):
    """Calculate the Quadratic Weighted Kappa (QWK)."""
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def train(model, train_loader, optimizer, scaler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch in tqdm(train_loader, desc="Training", leave=False):
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits.squeeze()

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = logits.round()

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            with autocast():
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                logits = outputs.logits.squeeze()

            total_loss += loss.item()
            # Round or convert predictions as necessary to ensure they are integers
            preds = logits.round().int()


            # Collect all predictions and labels for QWK calculation
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    qwk = calculate_qwk(np.array(all_labels), np.array(all_preds))
    accuracy = accuracy_score(np.array(all_labels), np.array(all_preds))
    return avg_loss, qwk, accuracy


access_token = "abc_dummy"

# Load your dataset
df = pd.read_csv('train.csv')

# Split the data into training and validation sets
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['score'])
train_scores = train_df['score'].values
valid_scores = valid_df['score'].values

# Initialize the tokenizer
# Since GPT-2 does not use padding during its initial training; it processes sequences end-to-end. So we will use EOF(end-of-sentence) token to handle the data.
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', token=access_token)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize essays
train_encodings = tokenizer(train_df['full_text'].tolist(), truncation=True, padding='max_length', max_length=512, pad_to_multiple_of=None)
test_encodings = tokenizer(valid_df['full_text'].tolist(), truncation=True, padding='max_length', max_length=512, pad_to_multiple_of=None)

train_dataset = EssayDataset(train_encodings, train_scores)
val_dataset = EssayDataset(test_encodings, valid_scores)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model_init().to(device)

scaler = GradScaler()

optimizer = Adam(model.parameters(), lr=5e-5)

scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
writer = SummaryWriter('runs/experiment_name')

# Main training loop
num_epochs = 20
train_losses = []
val_losses = []
train_qwks = []
val_qwks = []
val_accs = []

for epoch in range(num_epochs):
    train_loss, train_qwk = train(model, train_loader, optimizer, scaler, device)
    val_loss, val_qwk, val_accuracy = validate(model, val_loader, device)

    scheduler.step()
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('QWK/Train', train_qwk, epoch)
    writer.add_scalar('Loss/Val', val_loss, epoch)
    writer.add_scalar('QWK/Val', val_qwk, epoch)
    writer.add_scalar('Accuracy/Val', val_accuracy, epoch)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_qwks.append(train_qwk)
    val_qwks.append(val_qwk)
    val_accs.append(val_accuracy) 

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train QWK: {train_qwk:.4f}, Val Loss: {val_loss:.4f}, Val QWK: {val_qwk:.4f}, Val Accuracy: {val_accuracy:.4f}")

writer.close()
