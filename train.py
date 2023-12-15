'''
Created on 2023年10月17日

@author: sandy
'''
from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from PIL import Image
import collections
import numpy as np
import os
from keras import backend as K
import auto_game

# 将游戏截屏调整大小，归一化，并使其满足模型输入数据要求
def preprocess_images(image):
    # 图像大小调整为(80,80)，Image与numpy array互换时维度都会翻转
    # 如果宽度与高度不等时应特别注意
    x_t = np.array(Image.fromarray(image).resize((80, 80)))
    x_t = x_t.astype("float")
    x_t = (x_t - 50) / (213 - 50)  # 归一化
    # 形状调整为(1, 80, 80, 3)，以满足模型对输入数据的要求
    s_t = np.expand_dims(x_t, axis=0)
    return s_t

def get_next_batch(experience, model, num_actions, gamma, batch_size):
    # 从经验池中随机选取一批数据
    batch_indices = np.random.randint(low=0, high=len(experience), size=batch_size)
    batch = [experience[i] for i in batch_indices]
    # 训练数据初始化
    X = np.zeros((batch_size, 80, 80, 3))
    Y = np.zeros((batch_size, num_actions))
    # 给训练数据赋值
    for i in range(len(batch)):  # 第i个数据
        s_t, a_t, r_t, s_tp1, game_over = batch[i]
        X[i] = s_t  # 给第i个输入数据赋值
        Y[i] = model.predict(s_t)[0]  # (2,)
        if game_over:
            Y[i, a_t] = r_t  # 给第i个输出数据赋值
        else:
            Q_sa = np.max(model.predict(s_tp1)[0])
            Y[i, a_t] = r_t + gamma * Q_sa  # 给第i个输出数据赋值
    return X, Y

def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=8, strides=4,
                     kernel_initializer="normal",
                     padding="same",
                     input_shape=(80, 80, 3)))
    model.add(Activation("relu"))
    model.add(Conv2D(64, kernel_size=4, strides=2,
                     kernel_initializer="normal",
                     padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(64, kernel_size=3, strides=1,
                     kernel_initializer="normal",
                     padding="same"))
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dense(512, kernel_initializer="normal"))
    model.add(Activation("relu"))
    model.add(Dense(NUM_ACTIONS, kernel_initializer="normal"))
    model.compile(optimizer=Adam(learning_rate=1e-6), loss="mse")
    return model

# 定义常量，初始化变量
NUM_ACTIONS = 2  # 动作数量 (不动，向右移动)
GAMMA = 0.99  # 未来回报的折扣因子
INITIAL_EPSILON = 0.1  # epsilon的初值
FINAL_EPSILON = 0.0001  # epsilon的终值
epsilon = INITIAL_EPSILON
BATCH_SIZE = 32  # 批大小
num_wins = 0  # 初始化成绩
experience = collections.deque(maxlen=10000)  # 经验池
NUM_EPOCHS_OBSERVE = 100  # 观察轮数
NUM_EPOCHS_TRAIN = 500 # 训练轮数
NUM_EPOCHS = NUM_EPOCHS_OBSERVE + NUM_EPOCHS_TRAIN  # 总轮数
DATA_DIR = "C:\\Users\\sandy\\eclipse-workspace\\hit_ball\\data"
MODEL_PATH = os.path.join(DATA_DIR, "ball_hit_ball_model.h5")
TRAIN_LOG_PATH = os.path.join(DATA_DIR, "train_log.txt")

# 搭建和配置模型
model = get_model()
# 将模型保存到文件
model.save(MODEL_PATH, overwrite=True)
# 初始化游戏
game = auto_game.Ball_Hit_Ball_Game()

# 开始观察和训练
for e in range(NUM_EPOCHS):
    K.clear_session()  # 清除会话，防止内存耗尽
    model = load_model(MODEL_PATH)  # 重新载入模型
    fout = open(TRAIN_LOG_PATH, "a+")  # 打开训练记录文件
    loss = 0.0  # 初始化损失
    game.reset()  # 重置游戏状态

    # 获取第一个截屏
    a_0 = 0  # (0 = 不动, 1 = 右移动)
    x_t, r_0, game_over = game.step(a_0)
    # 对数据进行处理
    s_t = preprocess_images(x_t)

    while not game_over:
        s_tm1 = s_t  # 保存为t-1时刻的截屏
        # 确定一下个动作
        if e <= NUM_EPOCHS_OBSERVE:  # 观察期随机探索
            a_t = np.random.randint(low=0, high=NUM_ACTIONS, size=1)[0]
        else:  # 观察期以后
            if np.random.rand() <= epsilon:  # 随机探索
                a_t = np.random.randint(low=0, high=NUM_ACTIONS, size=1)[0]
            else:  # 根据模型决策
                q = model.predict(s_t)[0]
                a_t = np.argmax(q)
        x_t, r_t, game_over = game.step(a_t)  # 执行动作，获取反馈
        s_t = preprocess_images(x_t)  # 对数据进行处理
        if r_t == 1:  # 如果篮球碰到红球，记录成功一次
            num_wins += 1
        # 将原状态、动作、回报、新状态和游戏是否结束等信息保存到经验池中
        experience.append((s_tm1, a_t, r_t, s_t, game_over))
        if e > NUM_EPOCHS_OBSERVE:  # 观察期以后
            # 从经验集合中抽取一批数据
            X, Y = get_next_batch(experience, model, NUM_ACTIONS, GAMMA, BATCH_SIZE)
            loss += model.train_on_batch(X, Y)  # 训练模型，并记录损失

    # 逐渐减小探索频率，增加利用频率
    if epsilon > FINAL_EPSILON:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / NUM_EPOCHS

    # 保存训练数据
    fout.write("{:04d}\t{:.5f}\t{:d}\n".format(e + 1, loss, num_wins))
    fout.close()

    # 保存模型
    model.save(MODEL_PATH, overwrite=True)