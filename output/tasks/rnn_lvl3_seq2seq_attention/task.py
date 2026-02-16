"""Encoder-Decoder with Attention Mechanism for Sequence-to-Sequence Tasks."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error
import json
from pathlib import Path


class Seq2SeqDataset(Dataset):
    """Dataset for sequence-to-sequence tasks."""
    
    def __init__(self, src_sequences, tgt_sequences):
        self.src_sequences = src_sequences
        self.tgt_sequences = tgt_sequences
    
    def __len__(self):
        return len(self.src_sequences)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.src_sequences[idx], dtype=torch.long),
            torch.tensor(self.tgt_sequences[idx], dtype=torch.long)
        )


class Encoder(nn.Module):
    """Encoder module for sequence-to-sequence model."""
    
    def __init__(self, input_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


class Attention(nn.Module):
    """Attention mechanism for decoder."""
    
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        # hidden: (batch_size, hidden_size)
        # encoder_outputs: (batch_size, src_len, hidden_size)
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repeat hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Calculate attention energies
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        
        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    """Decoder module with attention for sequence-to-sequence model."""
    
    def __init__(self, output_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(output_size, embed_size)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden, cell, encoder_outputs):
        # x: (batch_size, 1)
        embedded = self.dropout(self.embedding(x))
        
        # Calculate attention weights
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        
        # Apply attention to encoder outputs
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        
        # Concatenate embedded input and context
        lstm_input = torch.cat((embedded, context), dim=2)
        
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        # Predict next token
        prediction = self.fc_out(output.squeeze(1))
        
        return prediction, hidden, cell, attn_weights


class Seq2Seq(nn.Module):
    """Sequence-to-Sequence model with attention."""
    
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.output_size
        
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # First decoder input is SOS token
        input_token = tgt[:, 0:1]
        
        for t in range(1, tgt_len):
            output, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            
            # Teacher forcing
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = output.argmax(1).unsqueeze(1)
            input_token = tgt[:, t:t+1] if teacher_force else top1
       
        return outputs


def generate_synthetic_data(num_samples=5000, max_len=10, vocab_size=20):
    """Generate synthetic sequence-to-sequence data."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    src_sequences = []
    tgt_sequences = []
    
    SOS = 0
    EOS = 1
    
    for _ in range(num_samples):
        # Generate source sequence (random integers)
        src_len = np.random.randint(3, max_len + 1)
        src = np.random.randint(2, vocab_size, src_len).tolist()
        
        # Generate target sequence (reverse of source for simple task)
        tgt = src[::-1]
        
        # Add SOS and EOS tokens
        src = [SOS] + src + [EOS]
        tgt = [SOS] + tgt + [EOS]
        
        src_sequences.append(src)
        tgt_sequences.append(tgt)
    
    # Pad sequences
    max_src_len = max(len(s) for s in src_sequences)
    max_tgt_len = max(len(t) for t in tgt_sequences)
    
    src_padded = np.array([s + [EOS] * (max_src_len - len(s)) for s in src_sequences])
    tgt_padded = np.array([t + [EOS] * (max_tgt_len - len(t)) for t in tgt_sequences])
    
    return src_padded, tgt_padded, max_src_len, max_tgt_len, vocab_size + 2


def split_data(X, y, train_ratio=0.8):
    """Split data into training and validation sets."""
    split_idx = int(len(X) * train_ratio)
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def train_epoch(model, dataloader, optimizer, criterion, device, clip=1.0):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0
    
    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        
        optimizer.zero_grad()
        
        output = model(src, tgt, teacher_forcing_ratio=0.5)
        
        # Reshape for loss calculation
        output = output[:, 1:].reshape(-1, output.shape[-1])
        tgt = tgt[:, 1:].reshape(-1)
        
        loss = criterion(output, tgt)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    epoch_loss = 0
    exact_matches = 0
    total_samples = 0
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            output = model(src, tgt, teacher_forcing_ratio=0)
            
            # Calculate loss
            output = output[:, 1:].reshape(-1, output.shape[-1])
            tgt_flat = tgt[:, 1:].reshape(-1)
            loss = criterion(output, tgt_flat)
            epoch_loss += loss.item()
            
            # Calculate exact match
            predictions = output.argmax(dim=1).reshape(src.shape[0], -1)
            targets = tgt[:, 1:]
            
            exact_matches += (predictions.cpu() == targets.cpu()).all(dim=1).sum().item()
            total_samples += src.shape[0]
    
    avg_loss = epoch_loss / len(dataloader)
    exact_match = exact_matches / total_samples
    
    return {
        'loss': avg_loss,
        'exact_match': exact_match
    }


def calculate_r2(model, dataloader, device):
    """Calculate R² score for regression-like evaluation."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            output = model(src, tgt, teacher_forcing_ratio=0)
            predictions = output.argmax(dim=2)
            
            all_preds.extend(predictions[:, 1:].flatten().cpu().numpy())
            all_targets.extend(tgt[:, 1:].flatten().cpu().numpy())
    
    if len(set(all_targets)) == 1:
        return 0.0
   
    return r2_score(all_targets, all_preds)


def main():
    """Main function to run the seq2seq attention task."""
    print("=" * 60)
    print("Encoder-Decoder with Attention - Sequence-to-Sequence Task")
    print("=" * 60)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate data
    print("\n1. Generating synthetic data...")
    X, y, max_src_len, max_tgt_len, vocab_size = generate_synthetic_data(
        num_samples=5000, max_len=8, vocab_size=15
    )
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Max source length: {max_src_len}, Max target length: {max_tgt_len}")
    
    # Split data
    X_train, X_val, y_train, y_val = split_data(X, y, train_ratio=0.8)
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Create datasets and dataloaders
    train_dataset = Seq2SeqDataset(X_train, y_train)
    val_dataset = Seq2SeqDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Model parameters
    embed_size = 128
    hidden_size = 256
    num_layers = 2
    dropout = 0.1
    
    # Initialize model
    print("\n2. Initializing model...")
    encoder = Encoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
    decoder = Decoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=1)  # Ignore EOS token
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("\n3. Training model...")
    num_epochs = 30
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), 'best_seq2seq_model.pt')
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Exact Match: {val_metrics['exact_match']:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load('best_seq2seq_model.pt'))
    
    # Evaluate on training data
    print("\n4. Evaluating on training data...")
    train_metrics = evaluate(model, train_loader, criterion, device)
    train_metrics['r2'] = calculate_r2(model, train_loader, device)
    print(f"Training Loss: {train_metrics['loss']:.4f}")
    print(f"Training Exact Match: {train_metrics['exact_match']:.4f}")
    print(f"Training R²: {train_metrics['r2']:.4f}")
    
    # Evaluate on validation data
    print("\n5. Evaluating on validation data...")
    val_metrics = evaluate(model, val_loader, criterion, device)
    val_metrics['r2'] = calculate_r2(model, val_loader, device)
    print(f"Validation Loss: {val_metrics['loss']:.4f}")
    print(f"Validation Exact Match: {val_metrics['exact_match']:.4f}")
    print(f"Validation R²: {val_metrics['r2']:.4f}")
    
    # Quality checks
    print("\n6. Quality checks...")
    assert val_metrics['exact_match'] > 0.95, f"Validation exact match should be > 0.95, got {val_metrics['exact_match']:.4f}"
    print(f"✓ Validation exact match: {val_metrics['exact_match']:.4f} > 0.95")
    
    assert val_metrics['r2'] > 0.9, f"Validation R² should be > 0.9, got {val_metrics['r2']:.4f}"
    print(f"✓ Validation R²: {val_metrics['r2']:.4f} > 0.9")
    
    assert val_metrics['loss'] < 1.0, f"Validation loss should be < 1.0, got {val_metrics['loss']:.4f}"
    print(f"✓ Validation loss: {val_metrics['loss']:.4f} < 1.0")
    
    # Save metrics
    print("\n7. Saving metrics...")
    metrics = {
        'train': {k: float(v) for k, v in train_metrics.items()},
        'validation': {k: float(v) for k, v in val_metrics.items()},
        'model_params': num_params,
        'vocab_size': vocab_size,
        'max_src_len': max_src_len,
        'max_tgt_len': max_tgt_len
    }
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics to metrics.json")
    
    print("\n" + "=" * 60)
    print("All quality checks passed!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
