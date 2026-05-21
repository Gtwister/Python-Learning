import multiprocessing as mp
import time


# 唱歌
def sing():
    for i in range(3):
        print('唱歌...')
        time.sleep(0.5)

# 跳舞
def dance():
    for i in range(3):
        print('跳舞...')
        time.sleep(0.5)

if __name__=='__main__':
    sing_process = mp.Process(target=sing)
    dance_process = mp.Process(target=dance)

    sing_process.start()
    dance_process.start()