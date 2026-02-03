"""
Continuous-Thinking Neural Network - Boredom Mechanism + Output Feedback

New features from community ideas:
1. Boredom mechanism: Confidence erodes over time, preventing "wait for cap" strategy
2. Output feedback: Network sees what it's outputting to improve self-awareness

Key insight: Make waiting costly, not just unrewarded.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import time


class BoredomConfidenceGate(nn.Module):
    """
    Confidence gate with boredom mechanism.
    
    Key innovation: Confidence erodes over time (boredom increases).
    Network can't game the system by waiting - boredom kills confidence.
    """
    def __init__(self, hidden_size, num_classes, threshold=0.70, boredom_rate=0.05):
        super(BoredomConfidenceGate, self).__init__()
        self.threshold = threshold
        self.boredom_rate = boredom_rate  # How fast boredom grows
        
        # Confidence estimator
        self.confidence_layer = nn.Linear(hidden_size, 1)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, num_classes)
        
    def forward(self, hidden_state, iteration, max_iterations):
        """
        Args:
            hidden_state: Current thought state
            iteration: Current iteration number
            max_iterations: Maximum iterations allowed
            
        Returns:
            output: Classification logits
            raw_confidence: Unadjusted confidence (TENSOR)
            adjusted_confidence: Confidence after boredom adjustment (TENSOR)
            should_output: Whether we're confident enough to speak (BOOL)
        """
        # Estimate raw confidence from hidden state
        confidence_logit = self.confidence_layer(hidden_state)
        raw_confidence = torch.sigmoid(confidence_logit).squeeze(-1)
        
        # Calculate boredom (grows with iterations)
        # Options: linear, exponential, sigmoid
        # Starting with linear for simplicity
        boredom = min(1.0, (iteration / max_iterations) * self.boredom_rate * 10)
        
        # Adjust confidence based on boredom
        # As boredom increases, confidence decreases
        adjusted_confidence = raw_confidence * (1 - boredom)
        
        # Generate output
        output = self.output_projection(hidden_state)
        
        # Check if we should output (based on adjusted confidence)
        # Detach for boolean logic, keep attached for return
        should_output = (adjusted_confidence.detach().mean() >= self.threshold)
        
        return output, raw_confidence, adjusted_confidence, should_output


class OutputFeedbackLayer(nn.Module):
    """
    Lets the network see what it's outputting.
    
    Key innovation: Network gets feedback about its own outputs,
    improving self-awareness and decision-making.
    """
    def __init__(self, num_classes, hidden_size):
        super(OutputFeedbackLayer, self).__init__()
        # Project output logits back into hidden space
        self.feedback_projection = nn.Linear(num_classes, hidden_size)
        
    def forward(self, output_logits):
        """
        Convert output logits into feedback signal.
        
        Args:
            output_logits: Current output predictions
            
        Returns:
            feedback: Signal to inject back into thinking loop
        """
        # Use softmax to get probability distribution
        output_probs = F.softmax(output_logits, dim=-1)
        
        # Project back to hidden space
        feedback = self.feedback_projection(output_probs)
        
        return feedback


class BoredomContinuousThinkingNet(nn.Module):
    """
    Continuous-thinking network with:
    - Boredom mechanism (confidence erodes over time)
    - Output feedback (network sees what it's outputting)
    """
    
    def __init__(self, input_size=784, hidden_size=256, num_classes=10,
                 confidence_threshold=0.70, max_iterations=50, 
                 num_recurrent_layers=3, boredom_rate=0.05):
        super(BoredomContinuousThinkingNet, self).__init__()
        
        self.hidden_size = hidden_size
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.num_recurrent_layers = num_recurrent_layers
        self.boredom_rate = boredom_rate
        
        # Input projection
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        # Multiple recurrent layers
        self.recurrent_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) 
            for _ in range(num_recurrent_layers)
        ])
        
        # Confidence gate with boredom
        self.confidence_gate = BoredomConfidenceGate(
            hidden_size, num_classes, confidence_threshold, boredom_rate
        )
        
        # Output feedback layer
        self.output_feedback = OutputFeedbackLayer(num_classes, hidden_size)
        
        # Feedback fusion (combine thinking with output feedback)
        self.feedback_fusion = nn.Linear(hidden_size * 2, hidden_size)
        
        # Hidden state
        self.hidden_state = None
        
    def reset_state(self):
        """Reset hidden state between different inputs"""
        self.hidden_state = None
        
    def forward(self, x, verbose=False, max_iterations=None):
        """
        Forward pass with boredom and output feedback.
        
        Args:
            x: Input tensor (batch_size, input_size)
            verbose: If True, print thinking process
            max_iterations: Override max iterations
            
        Returns:
            output: Classification logits
            raw_confidence: Confidence tensor (for loss)
            adjusted_confidence: Adjusted confidence tensor
            iterations: Number of thinking iterations
        """
        batch_size = x.size(0)
        
        # Use provided max_iterations or default
        iteration_limit = max_iterations if max_iterations is not None else self.max_iterations
        
        # Initialize hidden state
        if self.hidden_state is None or self.hidden_state.size(0) != batch_size:
            self.hidden_state = torch.zeros(batch_size, self.hidden_size).to(x.device)
        
        # Process input
        input_features = F.relu(self.input_layer(x))
        
        # Continuous thinking loop
        iterations = 0
        raw_confidence = torch.zeros(batch_size).to(x.device)
        adjusted_confidence = torch.zeros(batch_size).to(x.device)
        should_output = False
        output = None
        
        while iterations < iteration_limit:
            iterations += 1
            
            # === BRAIN: Continuous thinking ===
            
            # Combine current thoughts with input
            self.hidden_state = self.hidden_state + input_features
            
            # Recurrent processing
            current_state = self.hidden_state
            for recurrent_layer in self.recurrent_layers:
                current_state = F.relu(recurrent_layer(current_state))
            
            # === MOUTH: Confidence gate with boredom ===
            
            # Check if we should output (with boredom adjustment)
            output, raw_conf, adj_conf, should_output = self.confidence_gate(
                current_state, iterations, iteration_limit
            )
            raw_confidence = raw_conf
            adjusted_confidence = adj_conf
            
            # === OUTPUT FEEDBACK: Let network see what it's outputting ===
            
            # Get feedback from current output
            output_feedback = self.output_feedback(output)
            
            # Fuse thinking with output feedback
            combined = torch.cat([current_state, output_feedback], dim=-1)
            self.hidden_state = F.relu(self.feedback_fusion(combined))
            
            if verbose:
                boredom = min(1.0, (iterations / iteration_limit) * self.boredom_rate * 10)
                print(f"  Iteration {iterations}: Raw={raw_confidence.mean().item():.4f}, "
                      f"Adjusted={adjusted_confidence.mean().item():.4f}, Boredom={boredom:.4f}")
            
            # Output when adjusted confidence crosses threshold
            if should_output:
                if verbose:
                    print(f"  ✓ Confident! Outputting after {iterations} iterations")
                break
        
        if verbose and iterations == iteration_limit:
            print(f"  ⚠ Hit cap ({iteration_limit}). Raw={raw_confidence.mean().item():.4f}, "
                  f"Adjusted={adjusted_confidence.mean().item():.4f}")
        
        return output, raw_confidence, adjusted_confidence, iterations


def train_epoch_boredom(model, train_loader, optimizer, epoch, device, 
                        current_accuracy=0, current_threshold=0.70, current_cap=10):
    """
    Training epoch with correct incentives.
    
    Fixes:
    1. Added confidence_loss: Trains mechanism to actually estimate accuracy
    2. Inverted iteration penalty: Late answers cost MORE, not less
    3. Combined loss: Classification + Confidence * Penalty
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    total_iterations = 0
    cap_hits = 0
    total_raw_confidence = 0
    total_adjusted_confidence = 0
    
    # Dynamic iteration cap (monotonically increasing)
    if current_accuracy > 90 and current_threshold > 0.80:
        dynamic_cap = max(current_cap, 50)
    elif current_accuracy > 80:
        dynamic_cap = max(current_cap, 20)
    elif current_accuracy > 70:
        dynamic_cap = max(current_cap, 10)
    else:
        dynamic_cap = max(current_cap, 5)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        
        model.reset_state()
        optimizer.zero_grad()
        
        # Forward pass with boredom
        # Using dynamic cap
        output, raw_conf, adj_conf, iterations_used = model(
            data, verbose=False, max_iterations=dynamic_cap
        )
        
        # Track metrics
        total_iterations += iterations_used
        total_raw_confidence += raw_conf.mean().item()
        total_adjusted_confidence += adj_conf.mean().item()
        if iterations_used >= dynamic_cap:
            cap_hits += 1
        
        # 1. Classification Loss (Standard)
        class_loss = F.cross_entropy(output, target, reduction='none')
        
        # 2. Confidence Loss (New!)
        # Train confidence to match actual correctness (0 or 1)
        pred = output.argmax(dim=1)
        is_correct = pred.eq(target).float().detach() # Detach target!
        
        # MSE Loss for confidence (Raw confidence should match correctness)
        conf_loss = F.mse_loss(raw_conf, is_correct, reduction='none')
        
        # 3. Iteration Penalty (Inverted!)
        # Late answers cost MORE.
        # Penalty grows with iterations. E.g., 1.0 -> 1.5 -> 2.0
        # This encourages the network to minimize iterations by outputting early.
        # But it only outputs early if confidence is high.
        # So it must learn to be confident early.
        iteration_penalty = 1.0 + 0.1 * (iterations_used / dynamic_cap)
        
        # Combine losses
        # We value classification correctness AND calibrated confidence
        combined_loss = (class_loss + 2.0 * conf_loss) * iteration_penalty
        final_loss = combined_loss.mean()
        
        final_loss.backward()
        optimizer.step()
        
        total_loss += final_loss.item()
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                  f'Loss: {final_loss.item():.4f} '
                  f'Acc: {100. * correct / total:.1f}% '
                  f'Iters: {iterations_used} '
                  f'Raw: {raw_conf.mean().item():.3f} Adj: {adj_conf.mean().item():.3f}')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    avg_iterations = total_iterations / len(train_loader)
    cap_hit_rate = cap_hits / len(train_loader)
    avg_raw_conf = total_raw_confidence / len(train_loader)
    avg_adj_conf = total_adjusted_confidence / len(train_loader)
    
    print(f'\nEpoch {epoch} Summary:')
    print(f'  Accuracy: {accuracy:.2f}%')
    print(f'  Avg Loss: {avg_loss:.4f}')
    print(f'  Avg Iterations: {avg_iterations:.2f}')
    print(f'  Cap Hit Rate: {cap_hit_rate:.2%}')
    print(f'  Avg Raw Confidence: {avg_raw_conf:.4f}')
    print(f'  Avg Adjusted Confidence: {avg_adj_conf:.4f}')
    print(f'  Dynamic Cap: {dynamic_cap}')
    print(f'  Threshold: {model.confidence_gate.threshold:.4f}')
    print(f'  Boredom Rate: {model.boredom_rate:.4f}')
    
    # Ratchet up/down the threshold slightly to find the sweet spot?
    # No, keep it fixed to let the network adapt.
    
    return accuracy, avg_raw_conf, avg_adj_conf, cap_hit_rate, dynamic_cap


def test_boredom(model, test_loader, device):
    """Test with boredom mechanism"""
    model.eval()
    correct = 0
    total = 0
    total_iterations = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            
            model.reset_state()
            output, raw_conf, adj_conf, iterations = model(data, verbose=False)
            
            total_iterations += iterations
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    avg_iterations = total_iterations / len(test_loader)
    
    print(f'\nTest Results:')
    print(f'  Accuracy: {accuracy:.2f}%')
    print(f'  Avg Iterations: {avg_iterations:.2f}')
    
    return accuracy


if __name__ == "__main__":
    # Setup
    device = torch.device("cpu")
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Create model with boredom mechanism
    print("Creating Boredom-based Continuous Thinking Network...")
    print("Key features:")
    print("  1. Boredom mechanism: Confidence erodes over time")
    print("  2. Output feedback: Network sees what it's outputting")
    print("  3. Confidence loss: Network learns to predict correctness")
    print("  4. Iteration penalty: Late answers cost more")
    print()
    
    model = BoredomContinuousThinkingNet(
        input_size=784,
        hidden_size=256,
        num_classes=10,
        confidence_threshold=0.70,  
        max_iterations=50,
        num_recurrent_layers=3,
        boredom_rate=0.05 
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    current_accuracy = 0
    current_threshold = 0.70
    current_cap = 5
    
    print("Starting training with boredom mechanism...")
    print("="*60)
    
    for epoch in range(1, 6):
        accuracy, raw_conf, adj_conf, cap_rate, dynamic_cap = train_epoch_boredom(
            model, train_loader, optimizer, epoch, device,
            current_accuracy, current_threshold, current_cap
        )
        
        # Update for next epoch
        current_accuracy = accuracy
        current_cap = dynamic_cap
        
        # Test
        test_accuracy = test_boredom(model, test_loader, device)
        
        print("="*60)
    
    # Save model
    torch.save(model.state_dict(), 'boredom_continuous_thinking_net.pth')
    print("\nModel saved to 'boredom_continuous_thinking_net.pth'")
