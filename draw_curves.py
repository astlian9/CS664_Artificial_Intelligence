'''
Created on 2023年10月17日

@author: sandy
'''
import pandas as pd
import matplotlib.pyplot as plt

# 读取训练日志数据
with open('data\\train_log.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()

# 创建DataFrame并解析数据
df = pd.DataFrame(data)
df['epoch'], df['loss'], df['num_wins'] = zip(*df[0].str.split('\t').tolist())
df['epoch'] = df['epoch'].astype(int)
df['loss'] = df['loss'].astype(float)
df['num_wins'] = df['num_wins'].astype(int)

# 绘制训练过程损失曲线
plt.rcParams['figure.figsize'] = (6, 4)
plt.plot(df['epoch'], df['loss'])
plt.title('Training Loss Across Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

# 绘制训练过程得分曲线
plt.rcParams['figure.figsize'] = (6, 4)
plt.plot(df['epoch'], df['num_wins'])
plt.title('Number of Wins Across Epochs')
plt.ylabel('Num of Wins')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()