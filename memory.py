import numpy as np
import matplotlib.pyplot as plt
from aircraft_obj import AircraftEnv


def get_memory(groups_num):
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
    np.save('memory.npy', memory)


def glance_memory(v):
    memory = np.load('memory.npy')
    result = (memory[np.fabs(memory[:, 1] - v) < 5]).copy()
    return result


def small_memory():
    try:
        memory = np.load('memory.npy')
    except Exception:
        get_memory(300)
        memory = np.load('memory.npy')
    small_memory = memory[memory[:, -1] <= 10].copy()
    np.save('small_memory2.npy', small_memory)


if __name__ == '__main__':
    # get_memory(300)
    vs = [3000, 2500, 2000]
    result = glance_memory(vs[0])
    fig = plt.figure()
    plt.scatter(result[:, 0], result[:, 2])
    result = glance_memory(vs[1])
    plt.scatter(result[:, 0], result[:, 2])
    result = glance_memory(vs[2])
    plt.scatter(result[:, 0], result[:, 2])
    plt.legend(['v=%d' % vs[0], 'v=%d' % vs[1], 'v=%d' % vs[2]])
    plt.xlabel('w')
    plt.ylabel('range')
    plt.show()
    # small_memory()
