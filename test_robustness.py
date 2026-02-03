"""
Robustness Test Suite for Continuous Thinking Agent

Tests:
1. Adaptability to Speed: Can it solve 'Fast' inputs (5 steps) and wait patiently for 'Slow' inputs (30 steps)?
2. Robustness to Silence: Can it handle 50% blank periods without hallucinating?
"""

import torch
import numpy as np
from continuous_rl_env import ContinuousThinkingEnv
from rl_agent import ContinuousThinkingAgent
import time

def evaluate_agent(model, min_steps, max_steps, blank_prob, num_episodes=50):
    env = ContinuousThinkingEnv(min_steps=min_steps, max_steps=max_steps, blank_prob=blank_prob)
    device = torch.device("cpu") # Test on CPU
    model.to(device)
    model.eval()
    
    hidden_state = torch.zeros(1, model.hidden_size).to(device)
    state, _ = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    stats = {
        "correct": 0,
        "incorrect": 0,
        "timeout": 0,
        "hallucination": 0,
        "repeated_correct": 0,
        "thinking_steps": [],
        "total_digits": 0,
        "total_blanks": 0
    }
    
    current_content_steps = 0
    solved_current = False
    
    # Run for a fixed number of content switches
    contents_seen = 0
    
    # We need to hook into the env to know when content switches
    # But env switches internal state logic.
    # We can detect switch by checking if step_count reset (but we don't have access easily)
    # OR we just run for fixed variable 'global steps' and infer events from info dict.
    
    print(f"Running Test: Speed={min_steps}-{max_steps}, BlankProb={blank_prob:.1f}...")
    
    # Run loop
    max_test_steps = num_episodes * ( (min_steps + max_steps)/2 ) * 2 # Rough estimate
    
    step = 0
    while contents_seen < num_episodes:
        with torch.no_grad():
            action, _, _, new_hidden = model.get_action(state, hidden_state)
        
        obs, reward, done, _, info = env.step(action.item())
        state = torch.FloatTensor(obs).unsqueeze(0).to(device)
        hidden_state = new_hidden
        
        # Analyze Result
        res = info.get("result", None)
        
        # Check if content switched (heuristic: if reward is -2.0 (timeout) or just inferred?)
        # Environment was modified to NOT return 'switched' info.
        # But we know if we get "correct" or "incorrect", we are happy.
        # If we get -2.0 reward, it was a timeout.
        
        if reward == -2.0:
            stats["timeout"] += 1
            contents_seen += 1
            solved_current = False
        elif reward == -0.5:
            stats["hallucination"] += 1
            # Does blank switch on hallucination? No, persist.
        elif res == "correct":
            stats["correct"] += 1
            contents_seen += 1 # We consider it "done" for stats if solved? 
            # Ideally we count *Input Blocks*.
            # But the env is continuous.
            # Let's just track raw events.
            solved_current = True
        elif res == "incorrect":
            stats["incorrect"] += 1
            contents_seen += 1 
            solved_current = False
        elif res == "repeated_correct":
            stats["repeated_correct"] += 1
            
        # If blank phase ended? We can't easily see.
        # Let's rely on high-level stats over 2000 steps.
        step += 1
        if step > 2000: 
            break
            
    return stats

def run_suite():
    # Load Model
    model = ContinuousThinkingAgent(hidden_size=256)
    try:
        model.load_state_dict(torch.load("rl_agent.pth", map_location="cpu"))
        print("Loaded trained model.")
    except:
        print("Model not found, using random weights (Baseline).")

    # Test 1: Fast & Intense (No Blanks)
    # Simulates overwhelming information flow
    stats_fast = evaluate_agent(model, min_steps=5, max_steps=8, blank_prob=0.0)
    
    # Test 2: Slow & Boring (No Blanks)
    # Simulates plenty of time - does it wait?
    stats_slow = evaluate_agent(model, min_steps=20, max_steps=30, blank_prob=0.0)
    
    # Test 3: Sparse (Blanks)
    # Simulates need for silence
    stats_sparse = evaluate_agent(model, min_steps=10, max_steps=15, blank_prob=0.5)
    
    print("\n" + "="*60)
    print(f"{'Metric':<20} | {'Fast (5-8)':<15} | {'Slow (20-30)':<15} | {'Sparse (50% Blank)':<15}")
    print("-" * 60)
    
    metrics = ["correct", "incorrect", "timeout", "hallucination", "repeated_correct"]
    for m in metrics:
        v1 = stats_fast.get(m, 0)
        v2 = stats_slow.get(m, 0)
        v3 = stats_sparse.get(m, 0)
        print(f"{m:<20} | {v1:<15} | {v2:<15} | {v3:<15}")
    print("="*60)
    
    print("\nInterpretation:")
    print("- 'Correct' should be high in all non-sparse cases.")
    print("- 'Timeout' high in Fast means too slow thinking.")
    print("- 'Repeated' high in Slow means lack of patience.")
    print("- 'Hallucination' high in Sparse means inability to be silent.")

if __name__ == "__main__":
    run_suite()
