# write function that just stalls for one minute
import time

def stall():
    time.sleep(60)
    return

if __name__ == '__main__':
    print("test started")
    stall()
    print("test complete")