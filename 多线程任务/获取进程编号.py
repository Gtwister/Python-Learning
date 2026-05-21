import multiprocessing as mp
import time
import os


# 唱歌
def sing():
    print('唱歌进程的pid:',os.getpid())
    print('跳舞进程的父进程的pid:',os.getppid())
    for i in range(3):
        print('唱歌...')
        time.sleep(0.5)

# 跳舞
def dance():
    print('跳舞进程的pid:',os.getpid())
    print('跳舞进程的父进程的pid:',os.getppid())
    for i in range(3):
        print('跳舞...')
        time.sleep(0.5)

if __name__=='__main__':
    print('主进程的pid:',os.getpid())
    sing_process = mp.Process(target=sing)
    dance_process = mp.Process(target=dance)

    sing_process.start()
    dance_process.start()