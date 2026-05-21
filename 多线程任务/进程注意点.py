import time
import multiprocessing

def work():
    for i in range(10):
        print('工作中...')
        time.sleep(0.2)

if __name__=='__main__':
    # 创建子进程
    work_process = multiprocessing.Process(target = work)
    work_process.start()
    #让主进程等待1秒
    time.sleep(1)
    print('主进程执行完成了啦')