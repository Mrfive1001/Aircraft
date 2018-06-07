import numpy as np
import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
from memory import small_memory

# 定义神经网络并且获得网络结构
class DNN:
    def __init__(self, s_dim, a_dim, units, train=0, isnorm=True, name=None,graph = None):
        # 输入维度、输出维度、单元数、是否训练、名字
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.units = units
        self.train = train
        self.name = name
        # 对数据缩放
        self.scale = None
        self.isnorm = isnorm  # 控制是否缩放
        self.graph = graph
        # 保存网络位置
        self.model_path0 = os.path.join(sys.path[0], 'DNN_Net')
        if not os.path.exists(self.model_path0):
            os.mkdir(self.model_path0)
        if self.name is not None:
            self.model_path0 = os.path.join(self.model_path0, self.name)
            if not os.path.exists(self.model_path0):
                os.mkdir(self.model_path0)
        self.model_path = os.path.join(self.model_path0, 'data.chkp')
        if self.graph is None:
            self.graph = tf.get_default_graph()
        with self.graph.as_default():
            # 输入向量
            self.s = tf.placeholder(tf.float32, [None, s_dim], name='s')
            self.areal = tf.placeholder(tf.float32, [None, a_dim], 'areal')
            # 网络和输出向量
            net0 = tf.layers.dense(self.s, self.units, activation=tf.nn.relu, name='l0')
            net1 = tf.layers.dense(net0, self.units, activation=tf.nn.relu, name='l1')
            net2 = tf.layers.dense(net1, self.units, name='l2', activation=tf.nn.relu)
            net3 = tf.layers.dense(net2, self.units, name='l3', activation=tf.nn.relu)
            self.apre = tf.layers.dense(net3, self.a_dim, activation=tf.nn.tanh, name='apre')  # 输出线性

            self.loss = tf.reduce_mean(tf.squared_difference(self.areal, self.apre))  # loss函数
            self.train_op = tf.train.RMSPropOptimizer(0.001).minimize(self.loss)  # 训练函数
            # 保存或者读取网络
            if self.graph is not None:
                self.sess = tf.Session(graph=self.graph)
            else:
                self.sess = tf.Session(graph=self.graph)
            self.actor_saver = tf.train.Saver()
        if self.train == 1:
            self.sess.run(tf.global_variables_initializer())
        else:
            self.actor_saver.restore(self.sess, self.model_path)

    def learn(self, X, Y):
        # 使用网络对样本进行训练
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.s: X, self.areal: Y})
        return loss

    def store(self):
        # 存储网络
        self.actor_saver.save(self.sess, self.model_path)

    def predict(self, X):
        # 对X进行预测
        return self.sess.run(self.apre, feed_dict={self.s: X})

    def norm(self, data):
        # 将数据归1
        if self.isnorm:
            # 进行缩放
            if self.scale is None:
                self.scale = np.fabs(data).max(axis=0)
            return data / self.scale
        else:
            return data

    def unorm(self, data):
        # 将数据返归1
        if self.isnorm:
            return data * self.scale
        else:
            return data


if __name__ == '__main__':
    train_mode = 1  # 是否进行网络训练,0不训练，1从0开始训练，2从之前基础上开始训练
    num = 100
    net = DNN(2, 1, 256, train=train_mode, isnorm=True, name=str(num))  # 定义网络
    try:
        memory = np.load('Trajectories/memory_%s.npy'%num)  # 读取数据
    except Exception:
        memory = small_memory(num)
    memory_norm = net.norm(memory)
    if train_mode == 1 or train_mode == 2:
        # 训练模式
        X = memory_norm[:, 1:].copy()
        Y = memory_norm[:, 0:1].copy()
        losses = []
        for i in range(5000):
            sample_index = np.random.choice(len(X), size=5000)
            batch_x = X[sample_index, :]
            batch_y = Y[sample_index, :]
            loss = net.learn(batch_x, batch_y)
            losses.append(loss)
            print(i + 1, loss)
        plt.plot(losses)
        sample_index = np.random.choice(len(X), size=10)
        batch_x = X[sample_index, :]
        batch_y = Y[sample_index, :]
        batch_y_pre = net.predict(batch_x)
        print(batch_y, '\n\n')
        print(batch_y_pre - batch_y)
        net.store()
        plt.show()
    else:
        X = memory_norm[:, 1:].copy()
        Y = memory_norm[:, 0:1].copy()
        sample_index = np.random.choice(len(X), size=10)
        batch_x = X[sample_index, :]
        batch_y = Y[sample_index, :]
        batch_y_pre = net.predict(batch_x)
        print(batch_y, '\n\n')
        print(batch_y_pre - batch_y)
