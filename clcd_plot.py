import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from aircraft_obj import CAV

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 有中文出现的情况，需要u'内容'

# 气动参数测试

# 设置测试范围
alphas = np.arange(10, 20, 0.5)
mas = np.arange(0, 25, 0.5)
alphas, mas = np.meshgrid(alphas, mas)


def get_clcds(object):
    # 内部函数
    def get_cl(alpha, ma, object):
        cl, _ = object.alphama2clcd(alpha, ma)
        return cl

    def get_cd(alpha, ma, object):
        _, cd = object.alphama2clcd(alpha, ma)
        return cd

    cl_fun = np.frompyfunc(get_cl, 3, 1)  # 将自定义函数转化为通用函数3个输入，1个输出
    cd_fun = np.frompyfunc(get_cd, 3, 1)
    cls = cl_fun(alphas, mas, object)
    cds = cd_fun(alphas, mas, object)

    return cls, cds


def plot_craft(object):
    cls, cds = get_clcds(object)
    # fig是当前的图
    # ax是当前的轴
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title(u'升力系数')
    ax.set_ylabel(r'Mach')
    ax.set_xlabel(r'$\alpha$/$^\circ$')
    ax.invert_xaxis()  # 将x轴方向反转
    ax.plot_surface(alphas, mas, cds, rstride=1, cstride=1, cmap='rainbow')

    ax = fig.add_subplot(122, projection='3d')
    ax.set_title(u'阻力系数')
    ax.invert_xaxis()
    # ax.set_xlabel(r'$alpha$'+u'(度)')
    ax.set_ylabel(r'Mach')
    ax.set_xlabel(r'$\alpha$/$^\circ$')
    ax.plot_surface(alphas, mas, cls, rstride=1, cstride=1, cmap='rainbow')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.invert_xaxis()  # 将x轴方向反转
    ax.set_ylabel(r'Mach')
    ax.set_xlabel(r'$\alpha$/$^\circ$')
    cl_cds = cls / cds
    ax.plot_surface(alphas, mas, cl_cds, rstride=1, cstride=1, cmap='rainbow')


if __name__ == '__main__':
    # 低升阻比
    mycraft = CAV(low=True)
    plot_craft(mycraft)
    plt.show()
