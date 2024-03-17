import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
import time
from datetime import datetime
import sys
import os
from utils import Transition, ReplayMemory, VideoRecorder
from wrapper import AtariWrapper


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# get time
now = datetime.now()
time_format = "%Y%m%d_%H%M%S"
current_time = now.strftime(time_format)

if not os.path.exists("./boxing_{}".format(current_time)):
    os.makedirs("./boxing_{}".format(current_time))
log_file = "./boxing_{}/train_log".format(current_time)

noop_max = 30
frame_skip = 4
screen_size = 84
terminal_on_life_loss = True
clip_reward = True
action_repeat_probability = 0.0
normalize = True

# init env and model struct
env = gym.make('BoxingNoFrameskip-v4')
env = AtariWrapper(env, noop_max = noop_max, frame_skip = frame_skip, screen_size = screen_size, terminal_on_life_loss = terminal_on_life_loss, clip_reward = clip_reward, action_repeat_probability = action_repeat_probability, normalize = normalize)

env_str = (f"noop_max = {noop_max}, frame_skip = {frame_skip}, screen_size = {screen_size}, "
                  f"terminal_on_life_loss = {terminal_on_life_loss}, clip_reward = {clip_reward}, action_repeat_probability = {action_repeat_probability}, "
                  f"normalize = {normalize} \n")

print(env_str)
with open(log_file, mode='a') as filename:
    filename.write(env_str + '\n')

## training params
EPOCHS = 1000
EPISODE_LENGTH = 8192
TRAIN_BATCH_SIZE = 128
TRAIN_EPOCHS = 1
LEARNING_RATE = 0.00025
GRAD_CLIP = 5.0

## general params
NUM_ACTIONS = env.action_space.n
CRITIC_BETA = 0.5

## ppo params
CLIP_VALUE = 0.2
ENTROPY_BETA = 0.02

# gae params
USE_GAE = True
GAMA = 0.99
GAE_LAMBDA = 0.95


parameters_str = "start training + \n" + \
                 (f"GAMA = {GAMA}, GAE_LAMBDA = {GAE_LAMBDA}, EPISODE_LENGTH = {EPISODE_LENGTH}, "
                  f"TRAIN_BATCH_SIZE = {TRAIN_BATCH_SIZE}, TRAIN_EPOCHS = {TRAIN_EPOCHS}, LEARNING_RATE = {LEARNING_RATE}, "
                  f"CLIP_VALUE = {CLIP_VALUE}, ENTROPY_BETA = {ENTROPY_BETA}, NUM_ACTIONS = {NUM_ACTIONS}, "
                  f"CRITIC_BETA = {CRITIC_BETA}, GRAD_CLIP = {GRAD_CLIP}")

print(parameters_str)
with open(log_file, mode='a') as filename:
    filename.write(parameters_str + '\n')


class Logger(object):
    def __init__(self, output_file):
        self.terminal = sys.stdout
        self.log = open(output_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# init = tf.keras.initializers.Orthogonal()

# model class
class ActorModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=4, activation='relu', kernel_initializer=tf.keras.initializers.Orthogonal())
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=3, activation='relu', kernel_initializer=tf.keras.initializers.Orthogonal())
        self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer=tf.keras.initializers.Orthogonal())
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.Orthogonal())
        self.actor = layers.Dense(NUM_ACTIONS, activation='softmax', kernel_initializer=tf.keras.initializers.Orthogonal())
        self.critic = layers.Dense(1, kernel_initializer=tf.keras.initializers.Orthogonal())

    def forward(self, state):
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.actor(x), self.critic(x)

    def act(self, state, action=None):
        prob, v = self.forward(state)
        dist = tfp.distributions.Categorical(probs=prob)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'v': tf.squeeze(v)}  # 去除维度为1的维度

    def call(self, state, action = None):
        return self.act(state, action=action)

# ppo struct
class PPO:
    def __init__(self):
        self.model = ActorModel()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipvalue=GRAD_CLIP)
        # self.optimizer = tf.keras.optimizers.Adamax(learning_rate=LEARNING_RATE, clipvalue=GRAD_CLIP)

    def train(self, states, actions, advantages, log_probs_old, returns, values, clip_val, entropy_beta, idxes):
        total_loss = 0.0
        critic_loss = 0.0
        actor_loss = 0.0
        entropy_loss = 0.0

        ## get training sam data
        states = [states[i] for i in idxes]
        actions = [actions[i] for i in idxes]
        advantages = [advantages[i] for i in idxes]
        log_probs_old = [log_probs_old[i] for i in idxes]
        returns = [returns[i] for i in idxes]

        with tf.GradientTape() as tape:
            states = tf.expand_dims(states, 1)

            traj_info = self.model(states, actions)

            ## actor loss
            ratio = tf.exp(traj_info['log_pi_a'] - log_probs_old)
            advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
            advantages = tf.expand_dims(advantages, -1)
            surr1 = tf.multiply(ratio , advantages)
            surr2 = tf.multiply(tf.clip_by_value(ratio, 1 - clip_val, 1 + clip_val) , advantages)
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            # critic loss
            critic_loss = tf.keras.losses.mean_squared_error(returns, traj_info['v'])
            critic_loss = tf.reduce_mean(critic_loss)

            # entropy loss
            entropy_loss = tf.reduce_mean(traj_info['ent'])

            total_loss = actor_loss + CRITIC_BETA * critic_loss - entropy_beta * entropy_loss

            # backprop
            grads = tape.gradient(total_loss, self.model.trainable_variables)
            variables = self.model.trainable_variables
            self.optimizer.apply_gradients(zip(grads, variables))
            actor_grads_norm = tf.linalg.global_norm(grads)

        return critic_loss, actor_loss, entropy_loss, total_loss, actor_grads_norm

def gae(rewards, dones, values, epsoide_length, vals_last):
    returns = np.zeros_like(rewards)
    advantages = np.zeros_like(rewards)

    if not USE_GAE:
        for t in reversed(range(epsoide_length)):
            if t == epsoide_length - 1:
                returns[t] = rewards[t] + GAMA * (1 - dones[t]) * vals_last
            else:
                returns[t] = rewards[t] + GAMA * (1 - dones[t]) * returns[t + 1]
            advantages[t] = returns[t] - values[t]
    else:
        for t in reversed(range(epsoide_length)):
            if t == epsoide_length - 1:
                returns[t] = rewards[t] + GAMA * (1 - dones[t]) * vals_last
                td_error = returns[t] - values[t]
            else:
                returns[t] = rewards[t] + GAMA * (1 - dones[t]) * returns[t + 1]
                td_error = rewards[t] + GAMA * (1 - dones[t]) * values[t + 1] - values[t]
            advantages[t] = advantages[t] * GAE_LAMBDA * GAMA * (1 - dones[t]) + td_error
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
    return returns, advantages

def save_model(ppo, epoch):
    actor_save_path = f"./boxing_{current_time}/model_{epoch}"
    tf.keras.models.save_model(ppo.model, actor_save_path)

def evaluate_model(ppo, env, num_episodes=5):
    # state = env.reset()
    # state = np.array([state[0]])

    state = tf.convert_to_tensor(env.reset()[0])
    state = tf.stack([state, state, state, state], axis=-1)
    state = tf.expand_dims(state, 0)

    total_reward = []
    reward_ = 0
    for _ in range(num_episodes):
        done = False
        truncated = False
        while not (done or truncated):
            traj_info = ppo.model(state)
            action = traj_info['a'].numpy()
            next_state, reward, done, truncated, info = env.step(action[0])  # 更新状态信息
            reward_ += reward

            next_state = tf.expand_dims(next_state, 0)
            #next_state = tf.stack([next_state, state[:, :, :, 0], state[:, :, :, 1], state[:, :, :, 2]], axis=-1)
            next_state = tf.stack([state[:, :, :, 1], state[:, :, :, 2], state[:, :, :, 3], next_state], axis=-1)
            state = next_state

            if done or truncated:
                state = tf.convert_to_tensor(env.reset()[0])
                state = tf.stack([state, state, state, state], axis=-1)
                state = tf.expand_dims(state, 0)
                total_reward.append(reward_)
                reward_ = 0
                break

    average_reward = np.mean(total_reward)
    return average_reward

# def random_sample(inds, minibatch_size):
#     all_batches = [np.arange(i * minibatch_size, min((i + 1) * minibatch_size, inds)) for i in
#                    range((inds + minibatch_size - 1) // minibatch_size)]
#
#     perm_inds = np.random.permutation(len(all_batches))
#
#     for ind in perm_inds:
#         yield tf.convert_to_tensor(all_batches[ind], dtype=tf.int64)

def random_sample(inds, minibatch_size):
    inds = np.random.permutation(inds)
    num_full_batches = len(inds) // minibatch_size

    for i in range(num_full_batches):
        batch_inds = inds[i * minibatch_size: (i + 1) * minibatch_size]
        yield tf.convert_to_tensor(batch_inds, dtype=tf.int64)

    r = len(inds) % minibatch_size
    if r:
        yield tf.convert_to_tensor(inds[-r:], dtype=tf.int64)



state = tf.convert_to_tensor(env.reset()[0])
state = tf.stack([state, state, state, state], axis=-1)
state = tf.expand_dims(state, 0)

ppo = PPO()

start_time = time.time()
# training
for epoch in range(EPOCHS):
    states = []
    actions = []
    rewards = []
    dones = []
    values = []
    predictions = []
    done = False

    # collect data
    for episode in range(EPISODE_LENGTH):
        traj_info = ppo.model(state)

        log_prob_old = traj_info['log_pi_a']
        action = traj_info['a'].numpy()
        value = traj_info['v'].numpy()

        next_state, reward, done, _, _ = env.step(action[0])

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        values.append(value)
        predictions.append(log_prob_old)

        # next_state = tf.expand_dims(next_state, 0)
        # next_state = tf.stack([next_state, state[:, :, :, 0], state[:, :, :, 1], state[:, :, :, 2]], axis = -1)

        # 在这里我们保留state的后三个帧, 并加入新的观测帧next_state
        next_state = tf.expand_dims(next_state, 0)
        next_state = tf.stack([state[:, :, :, 1], state[:, :, :, 2], state[:, :, :, 3], next_state], axis=-1)
        state = next_state

        if done:
            state = tf.convert_to_tensor(env.reset()[0])
            state = tf.stack([state, state, state, state], axis=-1)
            state = tf.expand_dims(state, 0)
            # break


    traj_info_last = ppo.model(np.array(state))
    vals_last = traj_info_last['v'].numpy()
    # if done:
    #     vals_last = 0
    # else:
    #     traj_info_last = ppo.model(np.array(state))
    #     vals_last = traj_info_last['v'].numpy()

    returns, advantages = gae(rewards, dones, values, EPISODE_LENGTH, vals_last)

    # print(tf.convert_to_tensor(states).shape)

    states = tf.squeeze(states, [1])
    print("###################")

    # print(tf.convert_to_tensor(states).shape )

    ## train off policy
    for train_epoch in range(TRAIN_EPOCHS):
        for idxes in random_sample(EPISODE_LENGTH, TRAIN_BATCH_SIZE):
            critic_loss, actor_loss, entropy_loss, total_loss, grads_norm = ppo.train(states,
                                                                                       np.array(actions),
                                                                                       advantages,
                                                                                       np.array( predictions),
                                                                                       returns,
                                                                                       np.array(values[:-1]) ,
                                                                                       CLIP_VALUE ,
                                                                                       ENTROPY_BETA,
                                                                                       idxes)

    # CLIP_VALUE *= 0.999
    ENTROPY_BETA *= 0.999

    curr_time = time.time()
    log_msg = "train_epoch {} Epoch {}/{} completed. critic_loss = {}  actor_loss = {}  entropy_loss = {}  total_loss = {} grads_norm = {}  mins={:<10.2f}".format(
        train_epoch,
        epoch + 1,
        EPOCHS,
        critic_loss,
        actor_loss,
        entropy_loss,
        total_loss,
        grads_norm,
        float(curr_time - start_time) / 60
    )
    print(log_msg)
    with open(log_file, mode='a') as filename:
        filename.write(log_msg + '\n')

    # avg_reward = evaluate_model(ppo, env)
    # reward_msg = "Average reward at epoch {}: {}".format(epoch + 1, avg_reward)
    # print(reward_msg)
    # with open(log_file, mode='a') as filename:
    #     filename.write(reward_msg + '\n')
    #
    # ## only save good model
    # if avg_reward > 20:
    #     save_model(ppo, epoch + 1)


    if (epoch + 1) % 10 == 0:
        avg_reward = evaluate_model(ppo, env)
        reward_msg = "Average reward at epoch {}: {}".format(epoch + 1, avg_reward)
        print(reward_msg)
        with open(log_file, mode='a') as filename:
            filename.write(reward_msg + '\n')

        # ## only save good model
        # if epoch + 1 <= 100 or avg_reward > 20:
        save_model(ppo, epoch + 1)