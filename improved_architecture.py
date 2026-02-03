"""
Continuous-Thinking Neural Network - Improved Architecture

Key improvements:
1. Deeper recurrent processing (multiple layers)
2. Parallel attention mechanism (focus on relevant features)
3. Confidence gate layer (separates "brain" from "mouth")

The network thinks continuously in the hidden layers,
but only speaks when confident enough.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import time


class ConfidenceGate(nn.Module):
    """
    Separates "brain" from "mouth".
    Only passes signals to output when confidence threshold is met.
    """
    def __init__(self, hidden_size, num_classes, threshold=0.95):
        super(ConfidenceGate, self).__init__()
        self.threshold = threshold
        
        # Confidence estimator
        self.confidence_layer = nn.Linear(hidden_size, 1)
        
        # Output projection (only activated when confident)
        self.output_projection = nn.Linear(hidden_size, num_classes)
        
    def forward(self, hidden_state):
        """
        Args:
            hidden_state: Current thought state
            
        Returns:
            output: Classification logits
            confidence: Estimated confidence (0-1)
            should_output: Whether we're confident enough to speak
        """
        # Estimate confidence from hidden state
        confidence_logit = self.confidence_layer(hidden_state)
        confidence = torch.sigmoid(confidence_logit).squeeze(-1)
        
        # Generate output (always computed, but may not be "spoken")
        output = self.output_projection(hidden_state)
        
        # Check if we should output
        should_output = (confidence.mean() >= self.threshold)
        
        return output, confidence.mean().item(), should_output


class ParallelAttention(nn.Module):
    """
    Simple attention mechanism to focus on relevant features.
    Runs in parallel with recurrent processing.
    """
    def __init__(self, hidden_size):
        super(ParallelAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_state):
        """
        Self-attention over hidden state.
        Helps network focus on important features.
        """
        # Compute attention (simplified single-head)
        Q = self.query(hidden_state)
        K = self.key(hidden_state)
        V = self.value(hidden_state)
        
        # Attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(hidden_state.size(-1))
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        
        return attended


class ImprovedContinuousThinkingNet(nn.Module):
    """
    Improved continuous-thinking architecture with:
    - Deeper recurrent processing
    - Parallel attention
    - Confidence gate (brain/mouth separation)
    """
    
    def __init__(self, input_size=784, hidden_size=256, num_classes=10,
                 confidence_threshold=0.95, max_iterations=50, num_recurrent_layers=3):
        super(ImprovedContinuousThinkingNet, self).__init__()
        
        self.hidden_size = hidden_size
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.num_recurrent_layers = num_recurrent_layers
        
        # Input projection
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        # Multiple recurrent layers (deeper thinking)
        self.recurrent_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) 
            for _ in range(num_recurrent_layers)
        ])
        
        # Parallel attention mechanism
        self.attention = ParallelAttention(hidden_size)
        
        # Attention fusion (combine recurrent and attention paths)
        self.fusion = nn.Linear(hidden_size * 2, hidden_size)
        
        # Confidence gate (brain/mouth separation)
        self.confidence_gate = ConfidenceGate(hidden_size, num_classes, confidence_threshold)
        
        # Hidden state (persistent thought)
        self.hidden_state = None
        
    def reset_state(self):
        """Reset hidden state between different inputs"""
        self.hidden_state = None
        
    def forward(self, x, verbose=False, max_iterations=None):
        """
        Forward pass with improved continuous thinking.
        
        Args:
            x: Input tensor (batch_size, input_size)
            verbose: If True, print thinking process
            max_iterations: Override max iterations (for training variability)
            
        Returns:
            output: Classification logits
            confidence: Confidence score
            iterations: Number of thinking iterations
        """
        batch_size = x.size(0)
        
        # Use provided max_iterations or default
        iteration_limit = max_iterations if max_iterations is not None else self.max_iterations
        
        # Initialize hidden state if needed
        if self.hidden_state is None or self.hidden_state.size(0) != batch_size:
            self.hidden_state = torch.zeros(batch_size, self.hidden_size).to(x.device)
        
        # Process input into features (do this once, reuse in loop)
        input_features = F.relu(self.input_layer(x))
        
        # Continuous thinking loop
        iterations = 0
        confidence = 0.0
        should_output = False
        
        while iterations < iteration_limit:
            iterations += 1
            
            # === BRAIN: Continuous thinking ===
            
            # Combine current thoughts with input (continuous access!)
            self.hidden_state = self.hidden_state + input_features
            
            # Recurrent processing (multiple layers)
            current_state = self.hidden_state
            for recurrent_layer in self.recurrent_layers:
                current_state = F.relu(recurrent_layer(current_state))
            
            # Parallel attention processing
            attended_state = self.attention(self.hidden_state)
            
            # Fuse recurrent and attention paths
            combined = torch.cat([current_state, attended_state], dim=-1)
            self.hidden_state = F.relu(self.fusion(combined))
            
            # === MOUTH: Confidence gate ===
            
            # Check if we should output
            output, confidence, should_output = self.confidence_gate(self.hidden_state)
            
            if verbose:
                print(f"  Iteration {iterations}: Confidence = {confidence:.4f}")
            
            # Output only when confident
            if should_output:
                if verbose:
                    print(f"  ✓ Confident! Outputting after {iterations} iterations")
                break
        
        if verbose and iterations == iteration_limit:
            print(f"  ⚠ Max iterations reached ({iteration_limit}). Confidence = {confidence:.4f}")
        
        # Track if we hit the cap or output confidently
        hit_cap = (iterations == iteration_limit)
        
        return output, confidence, iterations, hit_cap


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs, 
                current_accuracy=0.0, current_avg_confidence=0.0):
    """Train for one epoch with ADAPTIVE curriculum (adjusts to actual performance)"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    total_iterations = 0
    total_confidence = 0.0
    cap_hits = 0  # Count how many times we hit the cap
    confident_outputs = 0  # Count how many times we output confidently
    
    # Exponentially weighted moving average for cap hit rate
    ema_cap_rate = 0.5  # Start at 50% assumption
    ema_alpha = 0.1  # Smoothing factor for EMA
    
    # ADAPTIVE curriculum: Adjust difficulty based on actual performance
    # Low accuracy → Easy (low caps, low threshold)
    # High accuracy → Hard (high caps, high threshold)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        
        model.reset_state()
        optimizer.zero_grad()
        
        # DYNAMIC ITERATION CAP (monotonically increasing)
        # Start low, increase as model gets more confident
        # Track maximum cap seen so far (ratchet mechanism)
        
        if not hasattr(model, 'max_iteration_cap'):
            model.max_iteration_cap = 5  # Start very low
        
        # Increase cap based on current accuracy and confidence
        # Better performance → More thinking time allowed
        if current_accuracy > 95 and current_avg_confidence > 0.70:
            target_cap = 50  # Full thinking time
        elif current_accuracy > 90 and current_avg_confidence > 0.60:
            target_cap = 30  # Substantial thinking time
        elif current_accuracy > 85 and current_avg_confidence > 0.50:
            target_cap = 20  # Moderate thinking time
        elif current_accuracy > 75:
            target_cap = 10  # Limited thinking time
        else:
            target_cap = 5   # Very limited (force quick decisions)
        
        # Monotonic increase: can only go up
        dynamic_cap = max(model.max_iteration_cap, target_cap)
        model.max_iteration_cap = dynamic_cap
        
        # Set difficulty label based on cap
        if dynamic_cap <= 10:
            difficulty = "EASY"
        elif dynamic_cap <= 25:
            difficulty = "MEDIUM"
        else:
            difficulty = "HARD"
        
        # CAP-HIT-DRIVEN ADAPTIVE threshold with MONOTONIC INCREASE
        # Key insight: Adjust threshold to maintain ~30% cap hit rate
        # RATCHET MECHANISM: Threshold can only increase, never decrease
        # This prevents backward progress and ensures continuous improvement
        
        target_cap_rate = 0.30  # Target 30% cap hits for optimal learning
        
        # Calculate current cap hit rate
        total_samples = cap_hits + confident_outputs
        if total_samples > 10:  # Need some samples before adjusting
            current_cap_rate = cap_hits / total_samples
            
            # Update exponentially weighted moving average
            ema_cap_rate = ema_alpha * current_cap_rate + (1 - ema_alpha) * ema_cap_rate
            
            # Adjust threshold based on cap hit rate
            current_threshold = model.confidence_gate.threshold
            
            if ema_cap_rate > target_cap_rate + 0.05:  # Too many cap hits
                # Lower threshold to make it easier to output
                # BUT: Only if we haven't already raised it (monotonic constraint)
                adaptive_threshold = current_threshold * 0.98  # Decrease by 2%
            elif ema_cap_rate < target_cap_rate - 0.05:  # Too few cap hits
                # Raise threshold to push for more thinking
                adaptive_threshold = current_threshold * 1.02  # Increase by 2%
            else:
                # In target range, keep stable
                adaptive_threshold = current_threshold
            
            # MONOTONIC CONSTRAINT: Threshold can only increase
            # Track maximum threshold seen so far
            if not hasattr(model, 'max_threshold_seen'):
                model.max_threshold_seen = 0.30  # Start very low
            
            # Only allow increases, never decreases
            adaptive_threshold = max(adaptive_threshold, model.max_threshold_seen)
            model.max_threshold_seen = adaptive_threshold
            
            # Clamp to reasonable range
            adaptive_threshold = max(0.30, min(0.90, adaptive_threshold))
        else:
            # First few samples: start VERY easy
            adaptive_threshold = 0.30  # Start low, will only increase
            if not hasattr(model, 'max_threshold_seen'):
                model.max_threshold_seen = 0.30
        
        model.confidence_gate.threshold = adaptive_threshold
        
        # Forward pass with dynamic constraints
        output, confidence, iterations, hit_cap = model(data, max_iterations=dynamic_cap)
        
        # Track cap hits vs confident outputs
        if hit_cap:
            cap_hits += 1
        else:
            confident_outputs += 1
        
        # Loss combines classification, confidence, and ITERATION-WEIGHTED GRADIENT
        classification_loss = criterion(output, target)
        
        # Encourage network to be confident when correct
        pred = output.argmax(dim=1)
        correct_mask = (pred == target).float()
        confidence_loss = -torch.mean(correct_mask * torch.log(torch.tensor(confidence + 1e-8)))
        
        # ITERATION-WEIGHTED GRADIENT
        # Key insight: Weight the loss based on how many iterations were used
        # Early decisions (low iterations) → Higher gradient weight
        # Late decisions (high iterations) → Lower gradient weight
        # This encourages the network to decide as early as possible
        
        # Exponential decay: weight = exp(-decay_rate * (iterations - 1))
        # iterations=1 → weight=1.0 (full gradient)
        # iterations=10 → weight≈0.37 (reduced gradient)
        # iterations=50 → weight≈0.007 (very small gradient)
        decay_rate = 0.1  # Controls how fast gradient weight decays
        iteration_weight = np.exp(-decay_rate * (iterations - 1))
        
        # Apply iteration weight to the loss
        # This means early decisions get stronger learning signals
        weighted_classification_loss = iteration_weight * classification_loss
        weighted_confidence_loss = iteration_weight * confidence_loss
        
        # Total loss with iteration weighting
        loss = weighted_classification_loss + 0.1 * weighted_confidence_loss
        
        # Backward pass with iteration-weighted gradients
        loss.backward()
        
        # Gradient clipping (prevent divergence)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        total_iterations += iterations
        total_confidence += confidence
        
        if batch_idx % 100 == 0:
            current_acc = 100. * correct / total
            avg_iters = total_iterations / (batch_idx + 1)
            avg_conf = total_confidence / (batch_idx + 1)
            total_samples = cap_hits + confident_outputs
            cap_ratio = 100. * cap_hits / total_samples if total_samples > 0 else 0
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item():.4f} | Acc: {current_acc:.2f}% | '
                  f'Iters: {avg_iters:.1f} | MaxCap: {dynamic_cap} | '
                  f'Threshold: {model.confidence_gate.threshold:.3f} | '
                  f'HitCap: {cap_ratio:.0f}% (EMA: {ema_cap_rate*100:.0f}%) | {difficulty}')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    avg_iterations = total_iterations / len(train_loader)
    avg_confidence = total_confidence / len(train_loader)
    
    return avg_loss, accuracy, avg_iterations, avg_confidence


def test(model, test_loader, device, verbose_samples=5):
    """Test the model"""
    model.eval()
    correct = 0
    total = 0
    total_iterations = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            
            model.reset_state()
            
            verbose = (batch_idx == 0 and verbose_samples > 0)
            output, confidence, iterations, hit_cap = model(data, verbose=verbose)
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            total_iterations += iterations
            
            if verbose:
                for i in range(min(verbose_samples, data.size(0))):
                    print(f"\nSample {i+1}: True={target[i].item()}, "
                          f"Pred={pred[i].item()}, "
                          f"Confidence={F.softmax(output[i], dim=0).max().item():.4f}")
    
    accuracy = 100. * correct / total
    avg_iterations = total_iterations / len(test_loader)
    
    print(f'\nTest Accuracy: {accuracy:.2f}% ({correct}/{total})')
    print(f'Average Thinking Iterations: {avg_iterations:.1f}')
    
    return accuracy, avg_iterations


def main():
    """Main training loop"""
    print("=" * 70)
    print("IMPROVED Continuous-Thinking Neural Network")
    print("=" * 70)
    print("\nArchitecture improvements:")
    print("  ✓ Deeper recurrent processing (3 layers)")
    print("  ✓ Parallel attention mechanism")
    print("  ✓ Confidence gate (brain/mouth separation)")
    
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 5
    hidden_size = 256
    confidence_threshold = 0.90  # Slightly lower for testing
    max_iterations = 50
    num_recurrent_layers = 3
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load MNIST
    print("\nLoading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create improved model
    print("\nInitializing Improved Continuous-Thinking Network...")
    model = ImprovedContinuousThinkingNet(
        input_size=784,
        hidden_size=hidden_size,
        num_classes=10,
        confidence_threshold=confidence_threshold,
        max_iterations=max_iterations,
        num_recurrent_layers=num_recurrent_layers
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop with adaptive curriculum
    print("\n" + "=" * 70)
    print("Training with ADAPTIVE CURRICULUM")
    print("Difficulty adjusts based on actual performance, not fixed schedule")
    print("=" * 70)
    
    prev_accuracy = 0.0
    prev_confidence = 0.0
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"Previous performance: Acc={prev_accuracy:.1f}%, Conf={prev_confidence:.3f}")
        print(f"{'='*70}")
        
        train_loss, train_acc, train_iters, train_conf = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, num_epochs,
            current_accuracy=prev_accuracy, current_avg_confidence=prev_confidence
        )
        
        prev_accuracy = train_acc
        prev_confidence = train_conf
        print(f'\nEpoch {epoch} Summary:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Train Accuracy: {train_acc:.2f}%')
        print(f'  Avg Thinking Iterations: {train_iters:.1f}')
        
        print(f'\nTesting epoch {epoch}...')
        test_acc, test_iters = test(model, test_loader, device, verbose_samples=3)
        print()
    
    # Save model
    print("\nSaving improved model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden_size': hidden_size,
        'confidence_threshold': confidence_threshold,
        'max_iterations': max_iterations,
        'num_recurrent_layers': num_recurrent_layers,
    }, 'improved_continuous_thinking_net.pth')
    print("Model saved!")
    
    print("\n" + "=" * 70)
    print("IMPROVED ARCHITECTURE TEST COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
