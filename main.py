import pygame
import time
import os
import numpy as np 
import matplotlib.pyplot as plt
from function import TrafficEnv, TrafficVisualizer
from agent import QLearner

def run_fixed_time(env, steps_per_episode=150, green_duration=60):
    total_reward = 0
    env.reset()

    for step in range(steps_per_episode):
        action = (step // green_duration) % 4
        next_state, reward = env.step(action)
        if next_state is None:
            break
        total_reward += reward

        if step % 5 == 0:
            env.render(action)
            time.sleep(0.01)

    return total_reward


def main():
    visualizer = TrafficVisualizer()
    env = TrafficEnv(visualizer)
    
    agent = QLearner(learning_rate=0.01, discount_factor=0.9, exploration_rate=1.0)

    is_presenting = False
    if os.path.exists("traffic_brain.pkl"):
        agent.load_model("traffic_brain.pkl")
        agent.epsilon = 0.0 
        print("Resuming with smart agent!")
        is_presenting = True 
        episodes = 200
    else:
        print("Starting PURE AI training from scratch...")
        episodes = 200 
 
    steps_per_episode = 150 
    rewards_history = []

    try:
        for episode in range(episodes):
            env.reset()
            state = env._get_simplified_state() 
            total_reward = 0
            
            if is_presenting:
                should_slow_down = True
            else:
                should_slow_down = (episode < 3) or (episode >= (episodes - 3))
            
            if is_presenting:
                print(f"--- Presentation Episode {episode + 1} ---")
            
            for step in range(steps_per_episode):
                action = agent.choose_action(state, env.current_green)
                
                step_result = env.step(action)
                if step_result[0] is None: break 
                
                next_simplified_state, reward = step_result
                
                if agent.epsilon > 0:
                    agent.update_q_value(state, action, reward, next_simplified_state)
                
                state = next_simplified_state
                total_reward += reward
                
                actual_light = env.current_green
                
                if should_slow_down:
                    if not env.render(actual_light): 
                        print("Simulation stopped by user.")
                        return 
                    time.sleep(0.01) 
                else:
                    if not env.render(actual_light):
                        print("Simulation stopped by user.")
                        return

            if agent.epsilon > 0.05:
                agent.epsilon *= 0.99
          
            rewards_history.append(total_reward)
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{episodes}: Reward = {total_reward:.2f} | Epsilon = {agent.epsilon:.2f}")

        agent.save_model("traffic_brain.pkl")

    except KeyboardInterrupt:
        print("Stopped manually.")
        agent.save_model("traffic_brain.pkl")
    
    #Compare to fixed one baseline with last 5 episode of the agent
    baseline_rewards = []
    for i in range(5):
        r = run_fixed_time(env, steps_per_episode=150, green_duration=60)
        baseline_rewards.append(r)

    baseline_avg = np.mean(baseline_rewards)
    agent_avg = np.mean(rewards_history[-5:]) 

    print("\n===== Baseline Comparison =====")
    print(f"Fixed-Time Controller Avg Reward: {baseline_avg:.2f}")
    print(f"Q-Learning Agent Avg Reward: {agent_avg:.2f}")
    print(f"Improvement: {agent_avg - baseline_avg:.2f}")

    visualizer.close()
    is_presenting = False
    
    if not is_presenting and len(rewards_history) > 0:
        print("Generating Learning Graph...")
        
        episodes_range = np.arange(1, len(rewards_history) + 1)

        plt.figure(figsize=(10, 5))
        plt.plot(episodes_range, rewards_history, marker='.', linestyle='-', linewidth=1, label='Reward per Episode')

        if len(rewards_history) >= 20:
            moving_avg = np.convolve(rewards_history, np.ones(20)/20, mode='valid')
            plt.plot(episodes_range[19:], moving_avg, linewidth=2, label='20-Ep Moving Avg')

        plt.title('Pure Q-Learning Performance')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward (Higher is Better)')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(6,4))
        plt.bar(['Fixed-Time', 'Q-Learning'], [baseline_avg, agent_avg], color=['orange','green'])
        plt.ylabel("Average Total Reward")
        plt.title("Traffic Controller Performance Comparison")
        plt.grid(axis='y')
        plt.show()


if __name__ == "__main__":
    main()