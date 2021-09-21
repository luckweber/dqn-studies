from collections import deque
from DQN_Agent import DQN_Agent
import gym
import highway_env
import tensorflow as tf
import argparse
import numpy as np
import cv2
from gym import spaces
from PIL import Image


INPUT_SHAPE = (80, 80)
WINDOW_LENGTH = 4

def process_state(state):
    state.ndim == 3
    img = Image.fromarray(state)
    img = img.resize(INPUT_SHAPE).convert('L')
    processed_state = np.array(img)
    processed_state.shape == INPUT_SHAPE
    processed_state.astype('uint8')
    return processed_state



def train(env, agent):
    episodes = 1000
    all_rewards = 0
    blend = 4
    total_time = 0
    batch_size = 8
    
    for e in range(episodes):
        done = False
        state = process_state(env.reset())
        total_reward = 0
        game_score = 0

        
        while not done:  
            total_time += 1
            
            if total_time % agent.update_rate == 0:
                agent.update_target_model()
            
            
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action) 
            
            next_state = process_state(next_state)
            
            env.render()  
            agent.remember(state, action, reward, next_state, done)
            
            
            
            state = next_state
            game_score += reward
            reward -= 1  # Punish behavior which does not accumulate reward
            total_reward += reward
                    
            if done:
                all_rewards += game_score
                print("episode: {}/{}, game score: {}, reward: {}, avg reward: {}, total time: {}"
                    .format(e+1, episodes, game_score, total_reward, all_rewards/(e+1), total_time))
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                
    env.close()   
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train')
    parser.add_argument('--env_name', default='highway-v0')
    
    args = parser.parse_args()
    
    config = {
        "observation": {
            "type": "GrayscaleObservation",
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "stack_size": 4,
            "observation_shape": (80, 80)
        }
    }

        
    env = gym.make(args.env_name)
    # env.configure(config)
    
    action_size = env.action_space.n
    # state_size = (88, 80, 1)
    state_size = (WINDOW_LENGTH,) + INPUT_SHAPE
    print(state_size)
    agent = DQN_Agent(state_size, action_size)
    
    if args.mode == "train":
        train(env, agent)
    
    
if __name__ == "__main__":
    main()
    