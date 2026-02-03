"""
PPO Agent for Continuous Thinking
Adaptation of the recurrent architecture for Actor-Critic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ContinuousThinkingAgent(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), hidden_size=512, action_size=11):
        super(ContinuousThinkingAgent, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Feature Extractor (Deeper CNN)
        # 28x28 input
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # -> 14x14
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)# -> 7x7
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dimensions:
        # L1: 28 -> Pool -> 14
        # L2: 14 -> Pool -> 7
        # L3: 7 -> Pool -> 3 (floor((7+2*0-2)/2) + 1 = 3) OR floor(7/2) = 3.
        
        flatten_size = 128 * 3 * 3
        self.fc_features = nn.Linear(flatten_size, hidden_size)
        
        # Recurrent Core (Thinking)
        self.core = nn.GRUCell(hidden_size, hidden_size)
        
        # Actor Head (Policy)
        self.actor = nn.Linear(hidden_size, action_size)
        
        # Critic Head (Value)
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, x, hidden_state):
        # x shape: (batch_size, 1, 28, 28)
        
        # CNN Feature Extraction
        x = self.pool(F.relu(self.conv1(x))) # -> 16 channels, 14x14
        x = self.pool(F.relu(self.conv2(x))) # -> 32 channels, 7x7
        x = self.pool(F.relu(self.conv3(x))) # -> 64 channels, 3x3
        
        x = x.view(x.size(0), -1) # Flatten
        features = F.relu(self.fc_features(x))
        
        # Recurrent update
        new_hidden_state = self.core(features, hidden_state)
        
        # Heads
        action_logits = self.actor(new_hidden_state)
        state_value = self.critic(new_hidden_state)
        
        return action_logits, state_value, new_hidden_state
    
    def get_action(self, x, hidden_state):
        logits, value, new_hidden = self(x, hidden_state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value, new_hidden

