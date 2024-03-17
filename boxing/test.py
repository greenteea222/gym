import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
from wrapper import AtariWrapper

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = tf.keras.models.load_model("./breakout_20240314_215651/model_100")

noop_max = 30
frame_skip = 4
screen_size = 84
terminal_on_life_loss = True
clip_reward = False
action_repeat_probability = 0.0
normalize = True

# init env and model struct
env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
env = AtariWrapper(env, noop_max=noop_max, frame_skip=frame_skip, screen_size=screen_size,
                   terminal_on_life_loss=terminal_on_life_loss, clip_reward=clip_reward,
                   action_repeat_probability=action_repeat_probability, normalize=normalize)

print(env.action_space)


state = tf.convert_to_tensor(env.reset()[0])
state = tf.stack([state, state, state, state], axis=-1)
state = tf.expand_dims(state, 0)

total_reward = []
reward_ = 0
for _ in range(20):
    done = False
    truncated = False
    cnt = 0
    while not (done or truncated):
        cnt += 1
        # traj_info = model(state)
        # action = traj_info['a'].numpy()
        # next_state, reward, done, truncated, info = env.step(action[0])  # 更新状态信息
        action = env.action_space.sample() # 采取一个动作
        next_state, reward, done, truncated, info = env.step(action) # 更新状态信息

        reward_ += reward

        next_state = tf.expand_dims(next_state, 0)
        next_state = tf.stack([next_state, state[:, :, :, 0], state[:, :, :, 1], state[:, :, :, 2]], axis=-1)

        print(tf.reduce_mean(next_state - state), action)

        state = next_state

        env.render()
        #         print(reward)

        if done or truncated:
            state = tf.convert_to_tensor(env.reset()[0])
            state = tf.stack([state, state, state, state], axis=-1)
            state = tf.expand_dims(state, 0)
            total_reward.append(reward_)

            print(reward_, done, truncated, cnt)

            reward_ = 0
            break

average_reward = np.mean(total_reward)

print(f"mean reward = {average_reward}")
env.close()