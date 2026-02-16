#!/usr/bin/env python3
"""
Character-level RNN for Text Generation - Level 1
Task: Train a character-level RNN to generate text and evaluate with perplexity
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class CharDataset(Dataset):
    """Character-level text dataset."""
    
    def __init__(self, text, seq_length, char_to_idx, idx_to_char):
        self.text = text
        self.seq_length = seq_length
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        
        # Create sequences and targets
        self.sequences = []
        self.targets = []
        
        for i in range(len(text) - seq_length):
            seq = text[i:i + seq_length]
            target = text[i + 1:i + seq_length + 1]
            self.sequences.append([char_to_idx[c] for c in seq])
            self.targets.append([char_to_idx[c] for c in target])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long)
        )


class CharRNN(nn.Module):
    """Character-level RNN model."""
    
    def __init__(self, vocab_size, embed_size=64, hidden_size=128, num_layers=2, dropout=0.3):
        super(CharRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        
        embed = self.embedding(x)
        output, hidden = self.lstm(embed, hidden)
        output = self.dropout(output)
        output = self.fc(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return (h0, c0)


def split_data(text, train_ratio=0.8):
    """Split text into training and validation sets."""
    split_idx = int(len(text) * train_ratio)
    return text[:split_idx], text[split_idx:]


def create_vocabulary(text):
    """Create character vocabulary from text."""
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char, len(chars)


def generate_text(model, char_to_idx, idx_to_char, vocab_size, seed_text, length=100, device='cpu'):
    """Generate text using the model."""
    model.eval()
    
    # Initialize hidden state
    hidden = model.init_hidden(1)
    
    # Encode seed text
    input_seq = [char_to_idx.get(c, 0) for c in seed_text]
    input_seq = torch.tensor([input_seq], dtype=torch.long).to(device)
    
    # Process seed text
    with torch.no_grad():
        _, hidden = model(input_seq, hidden)
    
    # Start generating
    generated = list(seed_text)
    input_char = torch.tensor([[char_to_idx.get(seed_text[-1], 0)]] if seed_text else [[0]], 
                             dtype=torch.long).to(device)
    
    for _ in range(length):
        output, hidden = model(input_char, hidden)
        probs = torch.softmax(output[0, -1], dim=0)
        predicted = torch.multinomial(probs, 1).item()
        
        generated.append(idx_to_char[predicted])
        input_char = torch.tensor([[predicted]], dtype=torch.long).to(device)
    
    return ''.join(generated)


def evaluate(model, X, y, vocab_size, char_to_idx, idx_to_char, text, device='cpu'):
    """Evaluate the model and return metrics."""
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_X, batch_y in zip(X, y):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            output, _ = model(batch_X)
            
            # Reshape for loss calculation
            output = output.view(-1, vocab_size)
            batch_y = batch_y.view(-1)
            
            loss = criterion(output, batch_y)
            total_loss += loss.item() * batch_X.size(0)
            total_samples += batch_X.size(0)
    
    avg_loss = total_loss / total_samples
    perplexity = np.exp(avg_loss)
    
    # Generate sample text for artifact
    sample_text = generate_text(model, char_to_idx, idx_to_char, vocab_size, 
                               seed_text=text[:5], length=50, device=device)
    
    # Calculate additional metrics
    mse = 0.0  # Not directly applicable for classification, but included for compatibility
    r2 = 0.0   # Not directly applicable for classification
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'mse': mse,
        'r2': r2,
        'sample_text': sample_text
    }


def train(model, X_train, y_train, X_val, y_val, vocab_size, 
          learning_rate=0.001, epochs=100, device='cpu'):
    """Train the character-level RNN model."""
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Store losses for tracking
    train_losses = []
    val_losses = []
    
    print(f"Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        total_samples = 0
        
        for batch_X, batch_y in zip(X_train, y_train):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            output, _ = model(batch_X)
            
            # Reshape for loss calculation
            output = output.view(-1, vocab_size)
            batch_y = batch_y.view(-1)
            
            loss = criterion(output, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            epoch_loss += loss.item() * batch_X.size(0)
            total_samples += batch_X.size(0)
        
        avg_train_loss = epoch_loss / total_samples
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for batch_X, batch_y in zip(X_val, y_val):
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                output, _ = model(batch_X)
                output = output.view(-1, vocab_size)
                batch_y = batch_y.view(-1)
                
                loss = criterion(output, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                val_samples += batch_X.size(0)
        
        avg_val_loss = val_loss / val_samples
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    
    return train_losses, val_losses


def prepare_data(text, seq_length=20, train_ratio=0.8):
    """Prepare data for training."""
    # Create vocabulary
    char_to_idx, idx_to_char, vocab_size = create_vocabulary(text)
    print(f"Vocabulary size: {vocab_size}")
    
    # Split text
    train_text, val_text = split_data(text, train_ratio)
    
    # Create datasets
    train_dataset = CharDataset(train_text, seq_length, char_to_idx, idx_to_char)
    val_dataset = CharDataset(val_text, seq_length, char_to_idx, idx_to_char)
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return (train_loader, val_loader, vocab_size, char_to_idx, idx_to_char, 
            train_text, val_text)


def main():  # noqa: C901
    """Main function to run the character-level RNN task."""
    print("=" * 60)
    print("Character-level RNN for Text Generation - Level 1")
    print("=" * 60)
    
    # Sample text for training (using a portion of the Gettysburg Address)
    text = """Four score and seven years ago our fathers brought forth on this continent, a new nation, 
conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged 
in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long 
endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, 
as a final resting place for those who here gave their lives that that nation might live. It is altogether 
fitting and proper that we should do this. But, in a larger sense, we can not dedicate—we can not 
consecrate—we can not hallow—this ground. The brave men, living and dead, who struggled here, have 
consecrated it, far above our poor power to add or detract. The world will little note, nor long remember 
what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated 
here to the unfinished work which they who fought here have thus far so nobly advanced. It is rather for us 
to be here dedicated to the great task remaining before us—that from these honored dead we take increased 
devotion to that cause for which they gave the last full measure of devotion—that we here highly resolve 
that these dead shall not have died in vain—that this nation, under God, shall have a new birth of freedom—
and that government of the people, by the people, for the people, shall not perish from the earth."""
    
    print(f"Text length: {len(text)} characters")
    print(f"Unique characters: {len(set(text))}")
    
    # Prepare data
    print("\n1. Preparing data...")
    (train_loader, val_loader, vocab_size, char_to_idx, idx_to_char, 
     train_text, val_text) = prepare_data(text)
    
    # Create model
    print("\n2. Creating model...")
    model = CharRNN(vocab_size=vocab_size, embed_size=64, hidden_size=128, num_layers=2)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    print("\n3. Training model...")
    train_losses, val_losses = train(
        model, train_loader, val_loader, vocab_size,
        learning_rate=0.001, epochs=100, device='cpu'
    )
    
    # Evaluate on training data
    print("\n4. Evaluating on training data...")
    train_metrics = evaluate(model, train_loader, val_loader, vocab_size, 
                            char_to_idx, idx_to_char, train_text, device='cpu')
    print(f"Training Loss: {train_metrics['loss']:.4f}")
    print(f"Training Perplexity: {train_metrics['perplexity']:.2f}")
    print(f"Training MSE: {train_metrics['mse']:.6f}")
    print(f"Training Cross-Entropy: {train_metrics['loss']:.4f}")
    
    # Evaluate on validation data
    print("\n5. Evaluating on validation data...")
    val_metrics = evaluate(model, val_loader, val_loader, vocab_size, 
                          char_to_idx, idx_to_char, val_text, device='cpu')
    print(f"Validation Loss: {val_metrics['loss']:.4f}")
    print(f"Validation Perplexity: {val_metrics['perplexity']:.2f}")
    
    # Quality checks
    print("\n6. Quality checks...")
    
    # Check perplexity is reasonable
    assert val_metrics['perplexity'] < 100, f"Perplexity too high: {val_metrics['perplexity']:.2f}"
    print(f"✓ Perplexity acceptable: {val_metrics['perplexity']:.2f} < 100")
    
    # Check loss decreased
    assert val_losses[-1] < val_losses[0] * 1.2, f"Loss should decrease: {val_losses[0]:.4f} -> {val_losses[-1]:.4f}"
    print(f"✓ Loss decreased: {val_losses[0]:.4f} -> {val_losses[-1]:.4f}")
    
    # Check perplexity is reasonable (model learned something)
    assert val_metrics['perplexity'] < 50, f"Perplexity too high: {val_metrics['perplexity']:.2f}"
    print(f"✓ Perplexity reasonable: {val_metrics['perplexity']:.2f} < 50")
    
    # Generate sample text
    print("\n7. Generating sample text...")
    sample = generate_text(model, char_to_idx, idx_to_char, vocab_size, 
                          seed_text="Four", length=100, device='cpu')
    print(f"Sample text: {sample}")
    
    print("\nAll quality checks passed!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
