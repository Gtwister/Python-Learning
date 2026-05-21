import multiprocessing as mp
import time


# 唱歌
def sing(num,name):
    for i in range(num):
        print(name)
        print('唱歌...')
        time.sleep(0.5)

# 跳舞
def dance(num2,name):
    for i in range(num2):
        print(name)
        print('跳舞...')
        time.sleep(0.5)

if __name__=='__main__':
     sing_process = mp.Process(target=sing,args=(4,'小明'))
     dance_process = mp.Process(target=dance,kwargs={'name':'小红',"num2": 2})

     sing_process.start()
     dance_process.start()