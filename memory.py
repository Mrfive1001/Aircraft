import numpy as np
import matplotlib.pyplot as plt
from aircraft_obj import AircraftEnv


def get_memory(groups_num, load=True):
    if load:
        memory = np.load('memory.npy')  # 读取
    else:
        memory = None  # 存储的记忆数据
    env = AircraftEnv(True)
    for i in range(groups_num):
        w = np.random.rand()
        # _,info =
        if memory is not None:  # 保存数据
            memory = np.vstack([memory, info['store'].copy()])
        else:
            memory = info['store'].copy()
        print('You\'ve stored %d pieces of data.' % (len(memory)))
    np.save('memory.npy', memory)


if __name__ == '__main__':
    cav = AircraftEnv()
    # get_memory(100, load=False)
    s, info = cav.hv_w(0.5)
    cav.plot(s)
    # plt.plot(s[:, 3], info['h_refs'] / 1000)
    plt.show()
