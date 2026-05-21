import time
import threading

def sing():
    for i in range(3):
        print('唱歌...')
        time.sleep(1)

# 跳舞
def dance():
    for i in range(3):
        print('跳舞...')
        time.sleep(1)

if __name__ == '__main__':
    sing_threading = threading.Thread(target=sing)
    dance_threading = threading.Thread(target=dance)

    sing_threading.start()
    dance_threading.start()