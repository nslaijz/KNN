import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
train_data = {'宝贝当家':[45,2,9,'喜剧片'],
              '美人鱼':[21,17,5,'喜剧片'],
              '澳门风云3':[54,9,11,'喜剧片'],
              '功夫熊猫3':[39,0,31,'喜剧片'],
              '谍影重重':[5,2,57,'动作片'],
              '叶问3':[3,2,65,'动作片'],
              '我的特工爷爷':[6,5,21,'动作片'],
              '奔爱':[7,46,4,'爱情片'],
              '夜孔雀':[9,38,8,'爱情片'],
              '代理情人':[9,38,2,'爱情片'],
              '新步步惊心':[8,34,17,'爱情片'],
              '伦敦沦陷':[2,3,55,'动作片'],
              }

#将训练数据封装为dataframe
train_df =pd.DataFrame(train_data).T
#设置表格列名
train_df.columns =['搞笑镜头','拥抱镜头','打斗镜头','电影类型']
#设置测试数据
test_data = {'唐人街探案':[23,3,17]}
#计算欧氏距离

def euclidean_distance(vec1,vec2):
    return np.sqrt(np.sum(np.square(vec1 - vec2)))

#设定K值
K=3
movie = '唐人街探案'
d=[]
for train_x in train_df.values[:,:-1]:
    test_x = np.array(test_data[movie])
    d.append(euclidean_distance(train_x,test_x))

dd = pd.DataFrame(train_df.values,index=d)

dd1 =pd.DataFrame(dd.sort_index())
print(dd1.values[:K,-1:].max())
