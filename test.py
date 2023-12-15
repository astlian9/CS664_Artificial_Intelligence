'''
Created on 2023年10月17日

@author: sandy
'''
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from PIL import Image
import numpy as np
import os
import auto_game
import pygame
import winsound

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

# 初始化常量
DATA_DIR = "data"
MODEL_PATH = os.path.join(DATA_DIR, "ball_hit_ball_model.h5")
BATCH_SIZE = 32
NUM_EPOCHS = 30

# 载入模型，当模型在不同版本的keras下训练时，可将compile设置为False
model = load_model(MODEL_PATH, compile=False) 
model.compile(optimizer=Adam(learning_rate=1e-6), loss="mse")

# 启动游戏
game = auto_game.Ball_Hit_Ball_Game()

# 初始化变量
num_games, num_wins = 0, 0

for e in range(NUM_EPOCHS):
    game.reset()   
    a_0 = 0  # 选择第一个动作
    x_0, r_0, game_over = game.step(a_0) # 执行第一个动作
    s_t = preprocess_images(x_0)    # 数据处理
    while not game_over:
        # 完全由模型确定下一个动作
        q = model.predict(s_t, verbose=0)[0]
        a_t = np.argmax(q)        
        # 执行动作，获取反馈信息
        x_t, r_t, game_over = game.step(a_t)
        s_t = preprocess_images(x_t)  # 数据处理      
        
        if r_t == 1:  # 更新成功计数
            num_wins += 1     
            winsound.Beep(3000, 100)            
    num_games += 1 
pygame.quit() # 退出游戏    
print("游戏次数: {:03d}, 成功次数: {:03d}".format(num_games, num_wins))