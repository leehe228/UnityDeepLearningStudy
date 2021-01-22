#-*-coding:utf-8-*-
#!/usr/bin/env python3

import numpy as np
import random
import datetime
import time
import tensorflow as tf
from collections import deque
from mlagents.envs import UnityEnvironment

""" DQN Parameters """ 
state_size = [128, 128, 3]
action_size = 121

load_model = False
train_mode = True

batch_size = 32 # Mini-Batch Size
mem_maxlen = 50000 # Replay Memory Max Length
discount_factor = 0.9 # Discount Factor
learning_rate = 0.00025 # Learning Rate

run_episode = 5000 # Training Episode
test_episode = 1000 # Testing Episode
start_train_episode = 3 # Train to get Replay Memory before Real Training

target_update_step = 1000 # Updating Target Network Interval
print_interval = 1 # Printing Interval
save_interval = 1000 # Saving Model Interval

# ε-greedy Epsilon Setting
epsilon_init = 1.0 # ε at start
epsilon_min = 0.1 # ε at least

# Now DateTime to Write Log
date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")

# Unity Env Path
game = "SmallOmok"
env_name = "../Build/" + game

# Model File Save and Load Path
save_path = "../saved_model/" + game + "/" + date_time + "_DQN"
load_path = "../saved_model" + game + "YYYYMMDD-HH-MM-SS_DQN" + "/model/model"

""" Model Class : Build CNN, Define Loss Function, Network Optimizer """
class Model():
    def __init__(self, model_name):
        self.input = tf.placeholder(shape=[None, state_size[0], state_size[1], state_size[2]], dtype=tf.float32)
        
        # Normalize Input Data between -1, 1
        self.input_normalize = (self.input - (255.0 / 2)) / (255.0 / 2)

        # Build Convolutional Neural Network 
        with tf.variable_scope(name_or_scope=model_name):
            # Convolutional Layer 1
            self.conv1 = tf.layers.conv2d(inputs=self.input_normalize, filters=32, activation=tf.nn.relu, kernel_size=[8, 8], strides=[4, 4], padding="SAME")
            # Convolutional Layer 2
            self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64, activation=tf.nn.relu, kernel_size=[4, 4], strides=[2, 2], padding="SAME")
            # Convolutional Layer 3
            self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=64, activation=tf.nn.relu, kernel_size=[3, 3], strides=[1, 1], padding="SAME")
            # Full Connected Layer 1 
            self.flat = tf.layers.flatten(self.conv3)
            # Full Connected Layer 2
            self.fc1 = tf.layers.dense(self.flat, 512, activation=tf.nn.relu)
            # Output Layer
            self.Q_Out = tf.layers.dense(self.fc1, action_size, activation=None)
        # Predict
        self.predict = tf.argmax(self.Q_Out, 1)
        # Target Q Network
        self.target_Q = tf.placeholder(shape=[None, action_size], dtype=tf.float32)

        # Calculate Loss Function Value and Train Network
        self.loss = tf.losses.mean_squared_error(self.target_Q, self.Q_Out)
        self.UpdateModel = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model_name)


""" DQNAgent Class : Deep-Q-Network Algorithm """
class DQNAgent():
    def __init__(self):
        # Fields
        self.model1 = Model("Q1")
        self.target_model1 = Model("target1")
        self.model2 = Model("Q2")
        self.target_model2 = Model("target2")

        self.memory1 = deque(maxlen=mem_maxlen)
        self.memory2 = deque(maxlen=mem_maxlen)

        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self.epsilon = epsilon_init

        self.Saver = tf.train.Saver()
        self.Summary, self.Merge = self.Make_Summary()

        if load_model == True:
            self.Saver.restore(self.sess, load_path)

    # ε-Greedy (Exploration or Exploitation)
    def get_action(self, state1, state2):
        if self.epsilon > np.random.rand():
            # Exploration
            random_action1 = np.random.randint(0, action_size)
            random_action2 = np.random.randint(0, action_size)
            return random_action1, random_action2

        else:
            # Exploitation
            predict1 = self.sess.run(self.model1.predict, feed_dict={self.model1.input: state1})
            predict2 = self.sess.run(self.model2.predict, feed_dict={self.model2.input: state2})
            return np.asscalar(predict1), np.asscalar(predict2)
    
    # Add Data to Replay Memory
    # (State, Action, Reward, Next State, Done)
    def append_sample(self, data1, data2):
        self.memory1.append((data1[0], data1[1], data1[2], data1[3], data1[4]))
        self.memory2.append((data2[0], data2[1], data2[2], data2[3], data2[4]))


    # Save Network Model
    def save_model(self):
        self.Saver.save(self.sess, save_path + "/model/model")

    # Train
    def train_model(self, model, target_model, memory, done):
        # Reduce ε
        if done:
            if self.epsilon > epsilon_min:
                self.epsilon -= 0.5 / (run_episode - start_train_episode)

        # Mini-Batch Data Sampling for Learning
        mini_batch = random.sample(memory, batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        # Extract Data
        for i in range(batch_size):
            states.append(mini_batch[i][0])
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states.append(mini_batch[i][3])
            dones.append(mini_batch[i][4])

        # Calculate Target Value (τ)
        target = self.sess.run(model.Q_Out, feed_dict={model.input: states})
        target_val = self.sess.run(target_model.Q_Out, feed_dict={target_model.input: next_states})

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + discount_factor * np.amax(target_val[i])

        # Train and Calculate Loss Function Value
        _, loss = self.sess.run([model.UpdateModel, model.loss], feed_dict={model.input: states, model.target_Q: target})
        
        return loss

    # Update Target Network
    def update_target(self, model, target_model):
        for i in range(len(model.trainable_var)):
            self.sess.run(target_model.trainable_var[i].assign(model.trainable_var[i]))

    # Set Data to log on Tensorboard
    def Make_Summary(self):
        self.summary_loss1 = tf.placeholder(dtype=tf.float32)
        self.summary_loss2 = tf.placeholder(dtype=tf.float32)
        self.summary_reward1 = tf.placeholder(dtype=tf.float32)
        self.summary_reward2 = tf.placeholder(dtype=tf.float32)

        tf.summary.scalar("loss1", self.summary_loss1)
        tf.summary.scalar("reward1", self.summary_reward1)
        tf.summary.scalar("loss2", self.summary_loss2)
        tf.summary.scalar("reward2", self.summary_reward2)

        Summary = tf.summary.FileWriter(logdir=save_path, graph=self.sess.graph)
        Merge = tf.summary.merge_all()

        return Summary, Merge

    # Write Summary on Tensorboard
    def Write_Summary(self, reward1, loss1, reward2, loss2, episode):
        self.Summary.add_summary(self.sess.run(self.Merge, feed_dict={self.summary_loss1: loss1, self.summary_reward1: reward1, self.summary_loss2: loss2, self.summary_reward2: reward2}), episode)


# MAIN : Training
if __name__ == "__main__":
    
    # Unity Env
    env = UnityEnvironment(file_name=env_name)

    # Unity Brains
    brain_name1 = env.brain_names[0]
    brain_name2 = env.brain_names[1]

    brain1 = env.brains[brain_name1]
    brain2 = env.brains[brain_name2]

    env_info = env.reset(train_mode=train_mode)

    # DQNAgent Object
    agent = DQNAgent()

    step = 0

    rewards1 = []
    losses1 = []
    rewards2 = []
    losses2 = []

    # Training For Loop
    for episode in range(run_episode + test_episode):
        if episode == run_episode:
            train_mode = False

        # Reset Unity Env
        env_info = env.reset(train_mode=train_mode)

        done = False

        # Reset State, Episode Rewards, Done of Agents
        state1 = 255 * np.array(env_info[brain_name1].visual_observations)
        episode_rewards1 = 0
        done1 = False

        state2 = 255 * np.array(env_info[brain_name2].visual_observations)
        episode_rewards2 = 0
        done2 = False

        # Run Training Episode
        while not done:
            step += 1

            # Decide Action and Apply on Unity Env
            action1, action2 = agent.get_action(state1, state2)
            env_info = env.step(vector_action = {brain_name1: [action1], brain_name2: [action2]})

            # Get Information from Agent1
            next_state1 = 255 * np.array(env_info[brain_name1].visual_observations)
            reward1 = env_info[brain_name1].rewards[0]
            episode_rewards1 += reward1
            done1 = env_info[brain_name1].local_done[0]

            # Get Information from Agent2
            next_state2 = 255 * np.array(env_info[brain_name2].visual_observations)
            reward2 = env_info[brain_name2].rewards[0]
            episode_rewards2 += reward1
            done2 = env_info[brain_name2].local_done[0]

            done = done1 or done2

            # Save Data to Replay Memory
            if train_mode:
                data1 = [state1, action1, reward1, next_state1, done1]
                data2 = [state2, action2, reward2, next_state2, done2]

                agent.append_sample(data1, data2)
            else:
                time.sleep(0.02)
                agent.epsilon = 0.0
            
            # Update States
            state1 = next_state1
            state2 = next_state2

            if episode > start_train_episode and train_mode:
                # Training with Model
                loss1 = agent.train_model(agent.model1, agent.target_model1, agent.memory1, done)
                loss2 = agent.train_model(agent.model2, agent.target_model2, agent.memory2, done)
                losses1.append(loss1)
                losses2.append(loss2)

                # Update Target Network
                if step % (target_update_step) == 0:
                    agent.update_target(agent.model1, agent.target_model1)
                    agent.update_target(agent.model2, agent.target_model2)

        rewards1.append(episode_rewards1)
        rewards2.append(episode_rewards2)

        # Print Out Log of Training, Write Reward, Loss Data on Tensorboard
        if episode % print_interval == 0 and episode != 0:
            print("step: {} / episode: {} / epsilon: {:.3f}".format(step, episode, agent.epsilon))
            print("reward1: {:.2f} / loss1: {:.4f} / reward2: {:.2f} / loss2: {:.4f}".format(np.mean(rewards1), np.mean(losses1), np.mean(rewards2), np.mean(losses2)))
            print("")

            agent.Write_Summary(np.mean(rewards1), np.mean(losses1), np.mean(rewards2), np.mean(losses2), episode)

            # Reset Episode Data
            rewards1 = []
            losses1 = []
            rewards2 = []
            losses2 = []

        # Save Network Model
        if episode % save_interval == 0 and episode != 0:
            agent.save_model()
            print("Model Saved at Episode {}.".format(episode))

    # Kill Unity Env
    env.close()

    # Done
