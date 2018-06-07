import numpy as np
import matplotlib.pyplot as plt
from aircraft_obj import AircraftEnv


def get_memory(groups_num):
    # 生成数据,生成一次即可
    memory = None  # 存储的记忆数据
    for i in range(groups_num):
        w = i / groups_num
        cav = AircraftEnv()
        s, info = cav.hv_w(w)
        if memory is not None:  # 保存数据
            memory = np.vstack([memory, info['store'].copy()])
        else:
            memory = info['store'].copy()
        print('Episode %d,You\'ve stored %d pieces of data.' % (i + 1, len(memory)))
    # memory = [w,v,range]
    np.save('Trajectories\memory_7000.npy', memory)


def glance_memory(v):
    # 得到某个速度下的记忆
    memory = np.load('Trajectories\memory_7000.npy')
    result = (memory[np.fabs(memory[:, 1] - v) < 5]).copy()
    return result


def small_memory(limit_range):
    # 将原有记忆进行处理，小于某个range值得记忆进行保存
    try:
        memory = np.load('Trajectories\memory_7000.npy')
    except Exception:
        get_memory(300)
        memory = np.load('Trajectories\memory_7000.npy')
    small_memory = memory[memory[:, -1] <= limit_range].copy()  # 射程限制
    np.save('Trajectories\memory_%s.npy' % limit_range, small_memory)
    return small_memory


if __name__ == '__main__':
    small_memory(200)

    # 画出某个速度下射程和w的关系
    # vs = [3000, 2500, 2000]
    # result = glance_memory(vs[0])
    # fig = plt.figure()
    # plt.scatter(result[:, 0], result[:, 2])
    # result = glance_memory(vs[1])
    # plt.scatter(result[:, 0], result[:, 2])
    # result = glance_memory(vs[2])
    # plt.scatter(result[:, 0], result[:, 2])
    # plt.legend(['v=%d' % vs[0], 'v=%d' % vs[1], 'v=%d' % vs[2]])
    # plt.xlabel('w')
    # plt.ylabel('range')
    # plt.show()
