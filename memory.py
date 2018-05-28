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


def test_memory(v):
    memory = np.load('memory.npy')
    result = (memory[np.fabs(memory[:, 1] - v) < 5]).copy()
    return result


if __name__ == '__main__':
    # get_memory(3000)
    result = test_memory(6000)
    fig = plt.figure()
    plt.scatter(result[:, 0], result[:, 2])
    result = test_memory(5000)
    plt.scatter(result[:, 0], result[:, 2])
    result = test_memory(4000)
    plt.scatter(result[:, 0], result[:, 2])
    plt.legend(['v=6000', 'v=5000', 'v=4000'])
    plt.xlabel('w')
    plt.ylabel('range')
    plt.show()
