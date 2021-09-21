import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from DQN_Agent import DQN_Agent
import gym
import highway_env
from stable_baselines3 import PPO
import numpy as np


def process_frame(frame):
    mspacman_color = np.array([210, 164, 74]).mean()
    img = frame[1:176:2, ::2]    # Crop and downsize
    img = img.mean(axis=2)       # Convert to greyscale
    img[img==mspacman_color] = 0 # Improve contrast by making pacman white
    img = (img - 128) / 128 - 1  # Normalize from -1 to 1.
    
    return np.expand_dims(img.reshape(88, 80, 1), axis=0)


def blend_images(images, blend):
    avg_image = np.expand_dims(np.zeros((88, 80, 1), np.float64), axis=0)

    for image in images:
        avg_image += image
        
    if len(images) < blend:
        return avg_image / len(images)
    else:
        return avg_image / blend
    

 
def train_highway():
    
    env = gym.make("highway-v0")

    
    state_size = (88, 80, 1)
    action_size = env.action_space.n

    agent = DQN_Agent(state_size, action_size)
    
    episodes = 10
    batch_size = 8
    skip_start = 90  # MsPacman-v0 waits for 90 actions before the episode begins
    total_time = 0   # Counter for total number of steps taken
    all_rewards = 0  # Used to compute avg reward over time
    blend = 4        # Number of images to blend
    done = False
    
    config = {
        "observation": {
            "type": "GrayscaleObservation",
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "stack_size": 4,
            "observation_shape": (80, 80)
        }
    }
    
    
    env.configure(config)

    
    
    for e in range(episodes):
        total_reward = 0
        game_score = 0
        # state = process_frame(env.reset())
        state = env.reset()
        images = deque(maxlen=blend)  # Array of images to be blended
        images.append(state)
        
        for skip in range(skip_start): # skip the start of each game
            env.step(0)
        
        for time in range(50):
            env.render()
            total_time += 1
            
            # Every update_rate timesteps we update the target network parameters
            if total_time % agent.update_rate == 0:
                agent.update_target_model()
                
            # Return the avg of the last 4 frames
            # state = blend_images(images, blend)
            
            # Transition Dynamics
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            # Return the avg of the last 4 frames
            # next_state = process_frame(next_state)
            images.append(next_state)
            # next_state = blend_images(images, blend)
            
            # Store sequence in replay memory
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            game_score += reward
            reward -= 1  # Punish behavior which does not accumulate reward
            total_reward += reward
            
            
            if done:
                all_rewards += game_score
                print("episode: {}/{}, game score: {}, reward: {}, avg reward: {}, time: {}, total time: {}"
                    .format(e+1, episodes, game_score, total_reward, all_rewards/(e+1), time, total_time))
                break
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                        
    env.close()
        
 
   
if __name__ == '__main__':
    train_highway()