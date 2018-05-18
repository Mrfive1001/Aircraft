import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from aircraft_obj import CAV

# 气动参数测试

# 设置测试范围
alphas = np.arange(0, 30, 0.5)
mas = np.arange(0, 20, 0.5)
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
    ax = fig.add_subplot(121,projection='3d')
    ax.invert_xaxis()   # 将x轴方向反转
    ax.plot_surface(mas, alphas, cds, rstride=1, cstride=1, cmap='rainbow', alpha=0.7)

    ax = fig.add_subplot(122,projection='3d')
    ax.invert_xaxis()
    ax.plot_surface(mas, alphas, cls, rstride=1, cstride=1, cmap='rainbow', alpha=0.7)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.invert_xaxis()  # 将x轴方向反转
    cl_cds = cls/cds
    ax.plot_surface(mas, alphas, cl_cds, rstride=1, cstride=1, cmap='rainbow', alpha=0.7)
    # TODO 加上标签，整理格式

if __name__ == '__main__':
    # 低升阻比
    mycraft = CAV(low=True)
    plot_craft(mycraft)
    plt.show()