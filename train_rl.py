
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from continuous_rl_env import ContinuousThinkingEnv
from rl_agent import ContinuousThinkingAgent
from torch.distributions import Categorical

# Hyperparameters
LR = 3e-4
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 4
BATCH_SIZE = 64
TRAIN_STEPS = 25000 # Reduced for final calibrated run
HIDDEN_SIZE = 512 # Upscaled for "Larger Brain"

def train():
    env = ContinuousThinkingEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    agent = ContinuousThinkingAgent(hidden_size=HIDDEN_SIZE).to(device)
    
    # Load Pre-trained Vision
    try:
        agent.load_state_dict(torch.load("pretrained_agent.pth", map_location=device))
        print("Loaded Pre-trained Vision Weights!")
    except FileNotFoundError:
        print("Warning: Pre-trained weights not found. Starting from scratch.")
        
    optimizer = optim.Adam(agent.parameters(), lr=LR)
    
    # Init hidden state
    hidden_state = torch.zeros(1, HIDDEN_SIZE).to(device)
    state, _ = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device) # (1, 1, 28, 28)
    
    print("Starting PPO Training...")
    
    global_step = 0
    try:
        while global_step < TRAIN_STEPS:
            
            # 1. Collect Trajectory
            states = []
            actions = []
            log_probs = []
            rewards = []
            values = []
            hiddens = [] # Store hidden states to detach/re-use if needed
            labels = [] # For aux loss
            
            # Rollout
            for _ in range(512): # Shorter rollout for faster feedback loop
                
                with torch.no_grad():
                    # Pass current state + hidden
                    action, log_prob, value, new_hidden = agent.get_action(state, hidden_state)
                
                next_state, reward, done, _, info = env.step(action.item())
                
                # Debug print once
                if global_step == 0:
                    print(f"State shape: {state.shape}")
                    
                # Store transition
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(value)
                hiddens.append(hidden_state)
                
                # Store label for aux loss (handle tensor vs int)
                lbl = info["label"]
                if isinstance(lbl, torch.Tensor):
                    lbl = lbl.item()
                labels.append(lbl)
                
                state = torch.FloatTensor(next_state).unsqueeze(0).to(device) # (1, 1, 28, 28)
                hidden_state = new_hidden 
                # Critical: Detach to limit BPTT horizon effectively to 1 step for now.
                # True BPTT would require retaining the graph, which fills memory fast.
                # Since the hidden state captures history, we rely on the Critic to estimate 
                # the long-term value of that history.
                
                global_step += 1
                if global_step % 1000 == 0:
                    print(f"Step {global_step}/{TRAIN_STEPS}")
            
            # 2. Compute Advantages (GAE)
            returns = []
            advantages = []
            next_value = 0 # Bootstrap with 0 at end of rollout (simplified)
            running_adv = 0
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    non_terminal = 1.0 # Continuous task
                    delta = rewards[t] + GAMMA * next_value * non_terminal - values[t].item()
                else:
                    non_terminal = 1.0
                    delta = rewards[t] + GAMMA * values[t+1].item() * non_terminal - values[t].item()
                
                running_adv = delta + GAMMA * 0.95 * non_terminal * running_adv
                advantages.insert(0, running_adv)
                returns.insert(0, running_adv + values[t].item())
                
            # 3. Optimize
            states_t = torch.cat(states)
            actions_t = torch.cat(actions)
            old_log_probs_t = torch.cat(log_probs)
            returns_t = torch.tensor(returns).to(device)
            advantages_t = torch.tensor(advantages).to(device)
            hiddens_t = torch.cat(hiddens) # (B, Hidden)
            labels_t = torch.tensor(labels).to(device)
            
            # Optimize for K epochs
            for _ in range(K_EPOCHS):
                # In a real implementation we'd do mini-batches here.
                # Doing full-batch update (512 steps) for simplicity/stability.
                
                # Re-evaluate actions
                curr_action_logits, curr_values, _ = agent(states_t, hiddens_t)
                probs = F.softmax(curr_action_logits, dim=-1)
                dist = Categorical(probs)
                curr_log_probs = dist.log_prob(actions_t)
                entropy = dist.entropy().mean()
                
                # Ratio
                ratio = torch.exp(curr_log_probs - old_log_probs_t)
                
                # Surrogate Loss
                surr1 = ratio * advantages_t
                surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages_t
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                value_loss = F.mse_loss(curr_values.squeeze(), returns_t)
                
                # Aux Loss (Supervised Classification)
                # Filter valid labels (>= 0)
                valid_mask = (labels_t >= 0)
                aux_acc = 0.0
                if valid_mask.sum() > 0:
                    # Logits 1-10 correspond to Digits 0-9
                    digit_logits = curr_action_logits[valid_mask, 1:] 
                    target_digits = labels_t[valid_mask]
                    aux_loss = F.cross_entropy(digit_logits, target_digits)
                    
                    # Calculate Accuracy
                    pred_digits = torch.argmax(digit_logits, dim=1)
                    aux_acc = (pred_digits == target_digits).float().mean().item()
                else:
                    aux_loss = torch.tensor(0.0).to(device)
                
                # Total loss
                # Give Aux loss high weight initially to force learning features
                loss = actor_loss + 0.5 * value_loss - 0.01 * entropy + 1.0 * aux_loss
                
                optimizer.zero_grad()
                loss.backward() # Retain graph? No, we re-forward each time.
                optimizer.step()
            
            # Logging
            if global_step % 2048 == 0:
                avg_reward = sum(rewards) / len(rewards)
                print(f"Update: Avg Reward: {avg_reward:.4f} | Value Loss: {value_loss.item():.4f} | Aux Loss: {aux_loss.item():.4f} | Aux Acc: {aux_acc:.4f}")
                if valid_mask.sum() > 0:
                     print(f"DEBUG Sample: Pred {pred_digits[0].item()} vs Label {target_digits[0].item()}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model...")
    finally:
        print("Training Complete!")
        # Save model
        torch.save(agent.state_dict(), "rl_agent.pth")
        print("Agent saved to rl_agent.pth")

if __name__ == "__main__":
    train()
