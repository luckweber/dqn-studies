import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute,Convolution2D
from keras.optimizers import Adam
import keras.backend as K

class DQN_Agent:
    #
    # Initializes attributes and constructs CNN model and target_model
    #
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        
        # Hyperparameters
        self.gamma = 1.0            # Discount rate
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_min = 0.1      # Minimal exploration rate (epsilon-greedy)
        self.epsilon_decay = 0.995  # Decay rate for epsilon
        self.update_rate = 1000     # Number of steps until updating the target network
        
        # Construct DQN models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.model.summary()

    #
    # Constructs CNN
    #
    def _build_model(self):
        model = Sequential()
        
        model.add(Permute((2, 3, 1), input_shape=self.state_size))

                
        model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(self.action_size))
        model.add(Activation('linear'))
        return model

    #
    # Stores experience in replay memory
    #
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    #
    # Chooses action based on epsilon-greedy policy
    #
    def act(self, state):
        # Random exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state)
        
        return np.argmax(act_values[0])  # Returns action using policy

    #
    # Trains the model using randomly selected experiences in the replay memory
    #
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        
        
        for state, action, reward, next_state, done in minibatch:
            
            if not done:
                target = (reward + self.gamma * np.amax(self.target_model.predict(next_state)))
            else:
                target = reward
                
            # Construct the target vector as follows:
            # 1. Use the current model to output the Q-value predictions
            target_f = self.model.predict(state)
            
            # 2. Rewrite the chosen action value with the computed target
            target_f[0][action] = target
            
            # 3. Use vectors in the objective computation
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    #
    # Sets the target model parameters to the current model parameters
    #
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
            
    #
    # Loads a saved model
    #
    def load(self, name):
        self.model.load_weights(name)

    #
    # Saves parameters of a trained model
    #
    def save(self, name):
        self.model.save_weights(name)