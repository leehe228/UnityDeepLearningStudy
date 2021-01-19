# 라이브러리 불러오기
import numpy as np
import random
import datetime
import time
import tensorflow as tf
from collections import deque
from mlagents.envs import UnityEnvironment

""" DQN 파라미터 값 설정 """ 
state_size = [128, 128, 3]
action_size = 121

load_model = False # 이전 학습된 모델 불러오기 여부
train_mode = True # 학습 모드 

batch_size = 32 # 미니 배치 학습 사이즈
mem_maxlen = 50000 # 리플레이 메모리 사이즈
discount_factor = 0.9 # 감가율
learning_rate = 0.00025 # 네트워크 학습 속도

run_episode = 5000 # 학습 에피소드 횟수
test_episode = 1000 # 테스트 에피소드 횟수
start_train_episode = 10 # 학습 전 리플레이 메모리 용 학습 횟수 **

target_update_step = 1000 # 타겟 네트워크 업데이트 주기
print_interval = 1 # 학습 상황 출력 주기
save_interval = 1000 # 모델 저장 주기

# exploration 확률
epsilon_init = 1.0 # epsilon 시작값
epsilon_min = 0.1 # epsilon 최솟값

date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")

# 유니티 환경 경로 
game = "SmallOmok"
env_name = "../Build/" + game

# 모델 저장 및 불러오기 경로
save_path = "../saved_model/" + game + "/" + date_time + "_DQN"
load_path = "../saved_model" + game + "YYYYMMDD-HH-MM-SS_DQN" + "/model/model"

""" Model Class 합성곱 신경망 정의 및 손실함수 설정, 네트워크 최적화 알고리즘 설정 """
class Model():
    def __init__(self, model_name):
        self.input = tf.placeholder(shape=[None, state_size[0], state_size[1], state_size[2]], dtype=tf.float32)
        
        # 입력을 -1 ~ 1 로 정규화S
        self.input_normalize = (self.input - (255.0 / 2)) / (255.0 / 2)

        # 합성곱 신경망 구축
        with tf.variable_scope(name_or_scope=model_name):
            self.conv1 = tf.layers.conv2d(inputs=self.input_normalize, filters=32, activation=tf.nn.relu, kernel_size=[8, 8], strides=[4, 4], padding="SAME")
            self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64, activation=tf.nn.relu, kernel_size=[4, 4], strides=[2, 2], padding="SAME")
            self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=64, activation=tf.nn.relu, kernel_size=[3, 3], strides=[1, 1], padding="SAME")
            
            self.flat = tf.layers.flatten(self.conv3)

            self.fc1 = tf.layers.dense(self.flat, 512, activation=tf.nn.relu)
            self.Q_Out = tf.layers.dense(self.fc1, action_size, activation=None)
        self.predict = tf.argmax(self.Q_Out, 1)

        self.target_Q = tf.placeholder(shape=[None, action_size], dtype=tf.float32)

        # 손실함수 값 계산 및 네트워크 학습 수행
        self.loss = tf.losses.huber_loss(self.target_Q, self.Q_Out)
        self.UpdateModel = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model_name)


""" DQNAgent Class DQN 알고리즘 함수 정의"""
class DQNAgent():
    def __init__(self):

        # 클래스 함수 위한 값 설정
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

    # Epsilon Greedy 기법에 따라 행동 결정
    def get_action(self, state1, state2):
        if self.epsilon > np.random.rand():
            # 랜덤하게 행동 결정
            random_action1 = np.random.randint(0, action_size)
            random_action2 = np.random.randint(0, action_size)

            return random_action1, random_action2
        else:
            # 네트워크 연산에 따라 행동 결정
            predict1 = self.sess.run(self.model1.predict, feed_dict={self.model1.input: [state1]}) #
            predict2 = self.sess.run(self.model2.predict, feed_dict={self.model2.input: [state2]}) #
            return np.asscalar(predict1), np.asscalar(predict2)
    
    # 리플레이 메모리에 데이터 추가
    # (상태, 행동, 보상, 다음 상태, 게임 종료 여부)
    def append_sample(self, data1, data2):
        self.memory1.append((data1[0], data1[1], data1[2], data1[3], data1[4]))
        self.memory2.append((data2[0], data2[1], data2[2], data2[3], data2[4]))


    # 네트워크 모델 저장
    def save_model(self):
        self.Saver.save(self.sess, save_path + "/model/model")

    # 학습 수행
    def train_model(self, model, target_model, memory, done):
        # Epsilon 값 감소
        if done:
            if self.epsilon > epsilon_min:
                self.epsilon -= 0.5 / (run_episode - start_train_episode)

        # 학습을 위한 미니 배치 데이터 샘플링
        mini_batch = random.sample(memory, batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for i in range(batch_size):
            states.append(mini_batch[i][0])
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states.append(mini_batch[i][3])
            dones.append(mini_batch[i][4])

        # 타겟값 계산
        target = self.sess.run(model.Q_Out, feed_dict={model.input: states})
        target_val = self.sess.run(target_model.Q_Out, feed_dict={target_model.input: next_states})

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + discount_factor * np.amax(target_val[i])

        # 학습 수행 및 손실함수 값 계산
        _, loss = self.sess.run([model.UpdateModel, model.loss], feed_dict={model.input: states, model.target_Q: target})
        
        return loss

    # 타겟 네트워크 업데이트
    def update_target(self, model, target_model):
        for i in range(len(model.trainable_var)):
            self.sess.run(target_model.trainable_var[i].assign(model.trainable_var[i]))

    # 텐서보드에 기록할 값 설정 및 데이터 기록
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

    def Write_Summary(self, reward1, loss1, reward2, loss2, episode):
        self.Summary.add_summary(self.sess.run(self.Merge, feed_dict={self.summary_loss1: loss1, self.summary_reward1: reward1, self.summary_loss2: loss2, self.summary_reward2: reward2}), episode)


# 메인 함수 DQN 알고리즘 진행
if __name__ == "__main__":
    
    # 유니티 환경 경로 설정
    env = UnityEnvironment(file_name=env_name)

    # 유니티 브레인 설정
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=train_mode)

    # DQNAgent를 agent로 정의
    agent = DQNAgent()

    step = 0

    rewards1 = []
    losses1 = []
    rewards2 = []
    losses2 = []

    # 게임 진행 반복문
    for episode in range(run_episode + test_episode):
        if episode == run_episode:
            train_mode = False

        # 유니티 환경 리셋 및 학습 모드 설정
        env_info = env.reset(train_mode=train_mode)

        done = False

        # 첫 번째 에이전트의 상태, episode_rewards, done 초기화
        state1 = np.uint8(255 * np.array(env_info[brain_name].visual_observations[0]))
        episode_rewards1 = 0
        done1 = False

        state2 = np.uint8(255 * np.array(env_info[brain_name].visual_observations[0]))
        episode_rewards2 = 0
        done2 = False

        # 에피소드 진행
        while not done:
            step += 1

            # 액션 결정 및 유니티 환경에 액션 적용
            action1, action2 = agent.get_action(state1, state2)
            env_info = env.step(vector_action={brain_name: [action1, action2]})

            # 첫 번째 에이전트에 대한 다음 상태, 보상, 게임 종료 정보 획득
            next_state1 = np.uint8(255 * np.array(env_info[brain_name].visual_observations[0]))
            reward1 = env_info[brain_name].rewards[0]
            episode_rewards1 += reward1
            done1 = env_info[brain_name].local_done[0]

            # 두 번째 에이전트에 대한 다음 상태, 보상, 게임 종료 정보 획득
            next_state2 = np.uint8(255 * np.array(env_info[brain_name].visual_observations[0]))
            reward2 = env_info[brain_name].rewards[1]
            episode_rewards2 += reward1
            done2 = env_info[brain_name].local_done[1]

            done = done1 or done2

            # 학습 모드인 경우 리플레이 메모리에 데이터 저장
            if train_mode:
                data1 = [state1, action1, reward1, next_state1, done1]
                data2 = [state2, action2, reward2, next_state2, done2]

                agent.append_sample(data1, data2)
            else:
                time.sleep(0.02)
                agent.epsilon = 0.0
            
            # 상태 정보 업데이트
            state1 = next_state1
            state2 = next_state2

            if episode > start_train_episode and train_mode:
                # 학습 수행
                loss1 = agent.train_model(agent.model1, agent.target_model1, agent.memory1, done)
                loss2 = agent.train_model(agent.model2, agent.target_model2, agent.memory2, done)
                losses1.append(loss1)
                losses2.append(loss2)

                # 타겟 네트워크 업데이트
                if step % (target_update_step) == 0:
                    agent.update_target(agent.model1, agent.target_model1)
                    agent.update_target(agent.model2, agent.target_model2)

        rewards1.append(episode_rewards1)
        rewards2.append(episode_rewards2)

        # 게임 진행 상황 출력 및 텐서보드에 보상과 손실함수값 기록
        if episode % print_interval == 0 and episode != 0:
            print("step: {} / episode: {} / epsilon: {:.3f}".format(step, episode, agent.epsilon))
            print("reward1: {:.2f} / loss1: {:.4f} / reward2: {:.2f} / loss2: {:.4f}".format(np.mean(rewards1), np.mean(losses1), np.mean(rewards2), np.mean(losses2)))
            print("")

            agent.Write_Summary(np.mean(rewards1), np.mean(losses1), np.mean(rewards2), np.mean(losses2), episode)

            rewards1 = []
            losses1 = []
            rewards2 = []
            losses2 = []

        # 네트워크 모델 저장
        if episode % save_interval == 0 and episode != 0:
            agent.save_model()
            print("Model Saved at Episode {}.".format(episode))

    env.close()
