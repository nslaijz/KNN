import pandas as pd
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#获取波士顿房价数据集
boston = load_boston()
#获取数据集特征（训练数据X）
X=boston.data
#获取数据集标记（label数据Y）
y=boston.target
#特征归一化到[0,1]范围内，提升模型收敛速度
X=MinMaxScaler().fit_transform(X)
#划分训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2020)

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    '''线性回归算法实现'''
    def _init_(self,alpha=0.1,epoch=5000,fit_bias=True):
        '''
        alpha:学习率，控制参数更新的幅度
        epoch:在整个训练集上训练迭代（参数更新）的次数
        fit_bias:是否训练偏置项参数
        '''
        self.alpha = alpha
        self.epoch = epoch
        #cost_record记录每一次的迭代的经验风险
        self.cost_record=[]
        self.fit_bias = fit_bias
    #预测函数
    def predict(self,X_test):
        '''
        X_test:m x n 的numpy二维数组
        '''
        #模型有偏置项参数时：为每个测试样本增加特征 x_0=1
        x_0=np.ones(X_test.shape[0])
        X_test = np.column_stack((x_0,X_test))
            #根据公式返回结果
        return np.dot(X_test,self.w)

    #模型训练，使用梯度下降更新参数（模型参数）
    def fit(self,X_train,y_train,cost_record):
         #训练偏置项参数时：为每个训练样本增加特征 x_0=1
        self.cost_record=cost_record
        x_0= np.ones(X_train.shape[0])
        X_train = np.column_stack((x_0,X_train))
        #训练样本数量
        m=X_train.shape[0]
        #样本特征维数
        n=X_train.shape[1]
        #初始模型参数
        self.w =np.ones(n)

        #模型参数迭代
        for i in range (5000):
            #计算训练样本预测值
            y_pred =np.dot(X_train,self.w)
            #计算训练集经验风险
            cost = np.dot(y_pred - y_train,y_pred-y_train)/(2*m)
            #记录训练集风险
            self.cost_record.append(cost)
            #参数更新
            self.w -= 0.1 / m*np.dot(y_pred - y_train,X_train)
        #保存模型
        self.save_model()
    #显示经验风险的收敛趋势图
    def polt_cost(self):
        plt.plot(np.arange(self.epoch),self.cost_record)
        plt.xlabel("epoch")
        plt.ylabel("cost")
        plt.show()

    #保存模型参数
    def save_model(self):
        np.savetxt("model.txt",self.w)
    #加载模型参数
    def load_model(self):
        self.w = np.loadtxt("model.txt")
# 3.模型的训练和预测
#实例化一个对象
model = LinearRegression()
#在训练集上训练
model.fit(X_train,y_train,[])
#在测试集上预测
y_pred = model.predict(X_test)
print('偏置参数：','ToDo')
print('特权参数：','ToDo')
print('预测结果：',y_pred[:5])
