# import requests
# import json
#
# def fun_httpget():
#     # date1={'wd': 'python'}
#     r = requests.get(url='http://192.168.1.4:5000/test')
#     r.text
#     print(r.text)
# fun_httpget()

from threading import Timer
import time
from apscheduler.schedulers.blocking import BlockingScheduler
# import sleep


# def printHello():
#     print("Hello")
#     print("当前时间戳是", time.time())
# def loop_func(func, second):
#     # 每隔second秒执行func函数
#     # while True:
#         timer = Timer(second, func)
#         timer.start()
#         timer.join()
t1 = time.time()
for i in range(3):
    # print(i)
    # scheduler = BlockingScheduler()
    # scheduler.add_job(printHello, 'interval', seconds=2)
    # scheduler.start()
    # loop_func(printHello, 5)
    time.sleep(1)
t2 = time.time()
print(t2-t1)