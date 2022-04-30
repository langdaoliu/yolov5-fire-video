# import paho.mqtt.client as mqtt
# host = "112.6.243.114"
# port = 1883
#
# def client():
#   client = mqtt.Client()
#   client.connect(host, port, 60)
#   client.publish("chat","hello liefyuan",2)
#   client.loop_forever()
# if __name__ == '__main__':
#   client()

import time
import paho.mqtt.client as mqtt
import json

HOST = "112.6.243.114"  # 服务器ip地址
PORT = 1883  # 服务器端口
USER = 'admin'  # 登陆用户名
PASSWORD = 'public'  # 用户名对应的密码


def on_connect(client, userdata, flags, rc):
  rc_status = ["连接成功", "协议版本错误", "无效的客户端标识", "服务器无法使用", "用户密码错误", "无授权"]
  print("connect：", rc_status[rc])

def call_mqtt(message):
  client = mqtt.Client()
  client.on_connect = on_connect  # 注册返回连接状态的回调函数
  client.username_pw_set(USER, PASSWORD)  # 如果服务器要求需要账户密码
  # client.will_set("test/die", "我死了", 0)  # 设置遗嘱消息
  client.connect(HOST, PORT, keepalive=60)  # 连接服务器
  # client.disconnect() #断开连接，不会触发遗嘱消息

  TOPIC_PUB = "/xiazhuang/fire/event"  # 发布主题
  message = json.dumps(message, ensure_ascii=False)
  MESSAGE = message  # 载荷
  client.publish(TOPIC_PUB, MESSAGE, qos=0)  # 发布消息
# time.sleep(3)