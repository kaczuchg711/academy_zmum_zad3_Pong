import sys
from google.colab import drive
drive.mount('/content/drive')
# uaktualnij poniższą ścieżkę
path_nb = r'/content/drive/My Drive/Colab Notebooks/ZMUM_2023/'
sys.path.append(path_nb)


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from IPython import display as ipythondisplay
import time
print('Numpy version:', np.__version__)
print('Tensorflow version:', tf.__version__)
print('Keras version:', tf.keras.__version__)

env = gym.make("CartPole-v0")
env.seed(1) # reproducible, since RL has high variance

print("Enviornment has observation space = {}".format(env.observation_space))

n_actions = env.action_space.n
print("Number of possible actions that the agent can choose from = {}".format(n_actions))

def create_cartpole_model():
    x = tf.keras.layers.Input(shape=(4,))
    d = tf.keras.layers.Dense(units=32, activation='relu')(x)
    d = tf.keras.layers.Dense(units=32, activation='relu')(d)
    out = tf.keras.layers.Dense(units=n_actions, activation='softmax')(d)
    return tf.keras.models.Model(inputs=x, outputs=out)

cartpole_model = create_cartpole_model()

def choose_action(model, observation):
  observation = observation.reshape([1, -1])
  prob_weights = model.predict(observation)
  action = np.random.choice(n_actions, size=1, p=prob_weights.flatten())[0]
  return action


class Memory:
    def __init__(self):
        self.clear()

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    def add_to_memory(self, new_observation, new_action, new_reward):
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)


memory = Memory()


def normalize(x):
  x -= np.mean(x)
  x /= np.std(x)
  return x

def discount_rewards(rewards, gamma=0.95):
  discounted_rewards = np.zeros_like(rewards)
  R = 0
  for t in reversed(range(0, len(rewards))):
      # update the total discounted reward
      R = R * gamma + rewards[t]
      discounted_rewards[t] = R
  return normalize(discounted_rewards)



from tensorflow.keras.optimizers import Adam

learning_rate = 1e-3
optimizer = Adam(learning_rate)


cartpole_model = create_cartpole_model()
cartpole_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

for i_episode in range(100):
  print('episode:', i_episode)
  # Restart the environment
  observation = env.reset()

  while True:
      # using our observation, take an action
      action = choose_action(cartpole_model, observation)
      next_observation, reward, done, info = env.step(action)
      # add to memory
      memory.add_to_memory(observation, action, reward)
      # is the episode over? did you crash or do so well that you're done?
      if done:
          # determine total reward and keep a record of this
          total_reward = sum(memory.rewards)
          print(total_reward, len(memory.actions))
          # initiate training - remember we don't know anything about how the agent is doing until it's crashed!
          cartpole_model.fit(np.vstack(memory.observations), np.vstack(memory.actions), epochs=1, batch_size=len(memory.observations), sample_weight=discount_rewards(memory.rewards))
          memory.clear()
          break
      # update our observatons
      observation = next_observation



def save_video_of_model(model, env_name, filename='agent.mp4'):
  import skvideo.io
  from pyvirtualdisplay import Display
  display = Display(visible=0, size=(40, 30))
  display.start()

  env = gym.make(env_name)
  obs = env.reset()
  shape = env.render(mode='rgb_array').shape[0:2]

  out = skvideo.io.FFmpegWriter(filename)

  done = False
  while not done:

      frame = env.render(mode='rgb_array')
      out.writeFrame(frame)
      action = model.predict(obs.reshape((1,-1))).argmax()
      obs, reward, done, info = env.step(action)
  out.close()
  print("Successfully saved into {}!".format(filename))

save_video_of_model(cartpole_model, "CartPole-v0")

from IPython.display import HTML
import io, base64
video = io.open('./agent.mp4', 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''
<video controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
</video>'''.format(encoded.decode('ascii')))

