import gym
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# Define the Pong agent class
class PongAgent:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.input_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.output_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model

    def choose_action(self, state):
        state = np.reshape(state, [1, -1])  # Reshape state with unknown dimension
        return np.argmax(self.model.predict(state)[0])

# Create the Pong environment
env = gym.make('Pong-v0', frameskip=4)  # Upgrade to version v4 and set frameskip

observation = env.reset()

# Initialize the agent
if isinstance(observation, tuple):
    input_size = observation[0].shape
else:
    input_size = observation.shape
input_size = np.prod(input_size)  # Flatten the observation shape
output_size = env.action_space.n
agent = PongAgent(input_size, output_size)

# Custom rendering function
def render_frame(env):
    plt.imshow(env.ale.getScreenRGB())
    plt.pause(0.001)

# Training loop
for episode in range(1000):
    episode_reward = 0
    done = False

    # Perform a noop action to initialize the environment
    observation, reward, done = env.step(0)[:3]

    while not done:
        render_frame(env)  # Custom rendering function

        action = agent.choose_action(observation)
        observation, reward, done = env.step(action)[:3]
        episode_reward += reward

        # Update the agent's model
        agent.model.fit(np.reshape(observation, [1, -1]), np.eye(output_size)[action:action+1], verbose=0)
    # Print the episode reward
    print(f"Episode {episode + 1}: Reward = {episode_reward}")

env.close()