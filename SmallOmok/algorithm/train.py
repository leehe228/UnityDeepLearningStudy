# 라이브러리 불러오기
import numpy as np
import random
import datetime
import time
import tensorflow as tf
from collections import deque
from mlagents.envs import UnityEnvironment


# 유니티 환경 경로 
game = "SmallOmok"
env_name = "../Build/" + game
