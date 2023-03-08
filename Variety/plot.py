from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

# # 设置事件文件路径和输出CSV文件路径
event_file_path = 'D:/CARLA_0.9.11/WindowsNoEditor/PythonAPI/examples/carla-Variety/test_highway_OMAC_gru_4/events.out.tfevents.1678186640.DESKTOP-6VPSU4Q.7112.0'
csv_file_path = 'D:/CARLA_0.9.11/WindowsNoEditor/PythonAPI/examples/carla-Variety/test_highway_OMAC_gru_4/data.csv'

# 创建EventAccumulator对象
event_acc = EventAccumulator(event_file_path)

# 加载事件文件
event_acc.Reload()

# 获取标签名称和标签值
tags = event_acc.Tags()['scalars']
tags = tags[0:2]
print(tags)
# print(event_acc.Scalars(tag=tags))
data = {tag: [] for tag in tags}

for tag in tags:
    for scalar_event in event_acc.Scalars(tag=tag):
        data[tag].append(scalar_event.value)

# 创建DataFrame对象
df = pd.DataFrame(data)

# 将DataFrame对象保存为CSV文件
df.to_csv(csv_file_path, index=True)

import seaborn as sns
import matplotlib.pyplot as plt

# 加载奖励数据
rewards = pd.read_csv(csv_file_path)

# 平滑处理奖励数据
smooth_rewards = rewards.rolling(window=1).mean()

# 计算奖励均值和标准差
mean_rewards = smooth_rewards.groupby('index').agg('mean')['reward/0_total_evaluate_rewards']
std_rewards = smooth_rewards.groupby('index').agg('std')['reward/0_total_evaluate_rewards']

# 绘制奖励曲线和阴影
sns.set(style='darkgrid')
plt.figure(figsize=(10, 6))
plt.title('Smoothed Reward Curve')
plt.xlabel('Episode')
plt.ylabel('Reward')
sns.lineplot(data=mean_rewards, color='b')
plt.fill_between(mean_rewards.index, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, color='b')
plt.show()