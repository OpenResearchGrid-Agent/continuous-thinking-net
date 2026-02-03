"""
Continuous Thinking Environment for Reinforcement Learning
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torchvision import datasets, transforms

class ContinuousThinkingEnv(gym.Env):
    """
    Environment where an agent receives a stream of MNIST digits.
    
    Action Space:
        0: THINK (Continue processing, refine hidden state)
        1: OUTPUT (Make a prediction and receive new input)
        2-11: Actually, we need to output label + decision logic.
              Let's make it simpler: 
              Discrete(11) where:
              0 = THINK
              1-10 = OUTPUT digit (0-9)
    
    Observation Space:
        - Current Image (1, 28, 28)
        - Internal State (Hidden Size) - handled by agent recursion usually, 
          but here we just give image to agent, agent manages hidden state.
        
    Rewards:
        +1.0 for Correct Output
        -1.0 for Incorrect Output
        -0.01 per step (Time Penalty / Thinking Cost)
        -2.0 for Missed Input (Simulated constraint: max time before input changes)
    """
    
    def __init__(self, data_root='./data', min_steps=5, max_steps=30, blank_prob=0.5):
        super(ContinuousThinkingEnv, self).__init__()
        
        # Load MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self.dataset = datasets.MNIST(data_root, train=True, download=True, transform=transform)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=True)
        self.data_iter = iter(self.dataloader)
        
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.blank_prob = blank_prob
        
        self.current_step_count = 0
        self.current_image = None
        self.current_label = None
        self.is_blank = False
        self.steps_until_switch = 0
        
        # Action: 0 = Think, 1-10 = Output Digit 0-9
        self.action_space = spaces.Discrete(11)
        
        # Observation: The image (1, 28, 28)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1, 28, 28), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._next_content()
        return self._get_obs(), {}
        
    def _next_content(self):
        """Switch to next content (either digit or blank)"""
        self.current_step_count = 0
        self.has_solved = False # Reset solved state
        
        # Decide duration for this block
        self.steps_until_switch = np.random.randint(self.min_steps, self.max_steps + 1)
        
        # Decide if blank or digit
        if np.random.random() < self.blank_prob:
            self.is_blank = True
            self.current_image = torch.zeros(1, 28, 28)
            self.current_label = -1 # No valid label
        else:
            self.is_blank = False
            try:
                # DataLoader gives (B, C, H, W). batch_size=1 => (1, 1, 28, 28).
                # We want just (C, H, W) => (1, 28, 28).
                batch_img, batch_lbl = next(self.data_iter)
                self.current_image = batch_img[0] 
                self.current_label = batch_lbl[0]
            except StopIteration:
                self.data_iter = iter(self.dataloader)
                batch_img, batch_lbl = next(self.data_iter)
                self.current_image = batch_img[0]
                self.current_label = batch_lbl[0]
    
    def _get_obs(self):
        return self.current_image.numpy()

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        info = {}
        
        self.current_step_count += 1
        
        info["label"] = self.current_label # For aux loss
        
        # Check if input duration expired (input changes/disappears externally)
        if self.current_step_count >= self.steps_until_switch:
            # If we were on a digit and didn't output correct solution even once, penalty
            # But wait, if we missed it completely?
            # User wants penalty for missed inputs.
            if not self.is_blank and not self.has_solved:
               reward = -2.0 
            
            # Switch content
            self._next_content()
            return self._get_obs(), reward, terminated, truncated, info

        # Handle Action
        if action == 0: # THINK
            reward = -0.01 # Small cost for thinking
            
        else: # OUTPUT
            if self.is_blank:
                # Outputting during blank is hallucinations/noise -> Penalty
                reward = -0.5 
                info["result"] = "hallucination"
            else:
                predicted_digit = action - 1
                if predicted_digit == self.current_label.item():
                    if not self.has_solved:
                        reward = 1.0 # First time solved
                        self.has_solved = True
                        info["result"] = "correct"
                    else:
                        # Already solved. User said "remove rewards".
                        # To prevent spamming (since Think cost is -0.01), 
                        # we must make this strictly worse than Thinking.
                        reward = -0.02 
                        info["result"] = "repeated_correct"
                else:
                    reward = -1.0
                    info["result"] = "incorrect"
            
            # In this robust version, we DO NOT switch content immediately using ._next_content()
            # The input persists. Agent must learn to stop outputting.
            # So we remove the self._next_content() call here.
            pass
            
        return self._get_obs(), reward, terminated, truncated, info

