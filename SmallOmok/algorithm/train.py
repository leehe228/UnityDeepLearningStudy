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
action_size = 128

load_model = False # 이전 학습된 모델 불러오기 여부
train_mode = True # 학습 모드 

batch_size = 32 # 미니 배치 학습 사이즈
mem_maxlen = 50000 # 리플레이 메모리 사이즈
discount_factor = 0.9 # 감가율
learning_rate = 0.00025 # 네트워크 학습 속도

run_episode = 2000 # 학습 에피소드 횟수
test_episode = 500 # 테스트 에피소드 횟수
start_train_episode = 500 # 학습 전 리플레이 메모리 용 학습 횟수

target_update_step = 1000 # 타겟 네트워크 업데이트 주기
print_interval = 10 # 학습 상황 출력 주기
save_interval = 500 # 모델 저장 주기

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


