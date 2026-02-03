import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from rl_agent import ContinuousThinkingAgent

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 1 # Quick pretraining
LR = 0.001

def pretrain():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Init Agent
    # Must match the architecture in train_rl.py (512 hidden)
    agent = ContinuousThinkingAgent(hidden_size=512).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    print("Pre-training Vision System...")
    
    agent.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Reset hidden state for each batch (stateless vision)
            # Use batch size from data
            batch_sz = data.size(0)
            hidden = torch.zeros(batch_sz, 512).to(device)
            
            optimizer.zero_grad()
            
            # Forward
            # Agent returns: logits, value, new_hidden
            # logits shape: (B, 11) where 0=Think, 1..10=Digits
            action_logits, _, _ = agent(data, hidden)
            
            # Digits are indices 1..10
            digit_logits = action_logits[:, 1:] 
            
            loss = criterion(digit_logits, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = torch.argmax(digit_logits, dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%")
                
    print(f"Pre-training Complete. Final Accuracy: {100.*correct/total:.2f}%")
    torch.save(agent.state_dict(), "pretrained_agent.pth")
    print("Saved to pretrained_agent.pth")

if __name__ == "__main__":
    pretrain()
