# import the necessary packages
# from yolov5 import Darknet
# from camera import LoadStreams, LoadImages
from utils.general import non_max_suppression, scale_coords, check_imshow
import torch.backends.cudnn as cudnn
from flask import Flask,render_template,Response
from models.common import DetectMultiBackend
from utils.torch_utils import select_device, time_sync
from video_config import cfg
import detect
from datetime import timedelta
import datetime
import base64
from utils.plots import Annotator, colors, save_one_box
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from pathlib import Path
import time
import torch
import json
import cv2
from mqtt_client import *
import gc
import os
from gevent import pywsgi

def imageToStr(image):
    image_byte=base64.b64encode(image)
    # print(type(image_byte))
    image_str=image_byte.decode('utf8') #byte类型转换为str
    # print(type(image_str))
    return image_str

def create_json():
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S')
    json =\
        {
            "method":"OnEventNotify",
            "params":{
                "ability":"event_fire",
                "events":[{
                    "data":{
                        "alarm":{
                            "cameraId":"2dede75576944dc48bf2c423df7dddc1",
                            "cameraName":"夏庄二号地块全景",
                            "cameraType":2,
                            "confirmStatus":0,
                            "direction":"北0.00°",
                            "distance":"1.005",
                            "eventName":"热成像_指挥中心",
                            "eventType":"192515",
                            "height":"23.1904001006",
                            "horizontalAngle":"0.0",
                            "id":"eb6a572b-ef4c-4b2a-8788-2450e3613f51",
                            "latitude":"36.237547",
                            "longitude":"120.446508",
                            "originPanpos":"0.0",
                            "originTiltpos":"0.0",
                            "picUrls":"7d58=9669i84-=o6c1ap33cbb9a6-f2e845aba*9e1==sp***131==1t4667036020=8l7*1522o7*097-=4*9e18od150l43-c275c8@@@@7d58=9669i84-=o9c1ap75cb*7o7=0980*1l1=2286317664t1==141***ps==1e9*aba548e2f-6a9b207-11*le1-od550c437c28",
                            "pitchAngel":"0.0",
                            "startTime":"1641376260378",
                            "status":0,
                            "towerId":"862b5ae2-887c-4b41-997c-490258d1a52a",
                            "towerLatitude":"36.25181666667",
                            "towerLongitude":"120.46465555556",
                            "towerName":"指挥中心",
                            "zoom":"0.0"}
                        ,"imageUrl":"/pic?7d58=9669i84-=o9c1ap75cb*7o7=0980*1l1=2286317664t1==141***ps==1e9*aba548e2f-6a9b207-11*le1-od550c437c28",
                        "sendTime":"2022-01-05T17:51:01.225+08:00",
                        "visiblePicUrl":"/pic?7d58=9669i84-=o6c1ap33cbb9a6-f2e845aba*9e1==sp***131==1t4667036020=8l7*1522o7*097-=4*9e18od150l43-c275c8"},
                    "eventId":"eb6a572b-ef4c-4b2a-8788-2450e3613f51",
                    "eventType":"169001001",
                    "happenTime":"2022-01-05T17:51:00.378+08:00",
                    "srcIndex":"2dede75576944dc48bf2c423df7dddc1",
                    "srcType":"camera",
                    "status":0,
                    "timeout":0}],
                "sendTime":"2022-01-05T17:51:01.225+08:00"}
        }
    return json
# initialize a flask object
server = Flask(__name__)
server.config["SEND_FILE_MAX_AGE_DEFAULT"] = timedelta(seconds=1)

# initialize the video stream and allow the camera sensor to warmup
# with open('yolov5_config.json', 'r', encoding='utf8') as fp:
#     opt = json.load(fp)
#     print('[INFO] YOLOv5 Config:', opt)

# Load model
device = select_device(cfg.device)
model = DetectMultiBackend(cfg.weights, device=device, dnn=cfg.dnn, data=cfg.data, fp16=cfg.half)
stride, names, pt = model.stride, model.names, model.pt
is_file = Path(cfg.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
is_url = cfg.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
webcam = cfg.source.isnumeric() or cfg.source.endswith('.txt') or (is_url and not is_file)
# darknet = Darknet(opt)
if webcam:
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(cfg.source, img_size=cfg.imgsz, stride=stride, auto=pt)
else:
    dataset = LoadImages(cfg.source, img_size=cfg.imgsz, stride=stride, auto=pt)

@server.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

def detect_gen(dataset, feed_type):
    # view_img = check_imshow()
    # t0 = time.time()
    flag = True
    count = 0
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] = t2 - t1
        pred = model(im,augment=cfg.augment)
        t3 = time_sync()
        dt[1] = t3 - t2
        pred = pred.float()
        #nms
        pred = non_max_suppression(pred, cfg.conf_thres, cfg.iou_thres,  agnostic = cfg.agnostic_nms, max_det = cfg.max_det)
        dt[2] = time_sync() - t3
        # pred = non_max_suppression(pred, darknet.opt["conf_thres"], darknet.opt["iou_thres"])
        img_json ={}
        pred_boxes = []
        # json_data = []
        json_data = create_json()
        for i, det in enumerate(pred):
            seen += 1
            if webcam:  # batch_size >= 1
                feed_type_curr, p, s, im0, frame = "Camera_%s" % str(i), path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                feed_type_curr, p, s, im0, frame = "Camera", path, '', im0s, getattr(dataset, 'frame', 0)

            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=4, example=str(names))
            if det is not None and len(det):
                count = 0
                if flag:
                    call_mqtt(json_data)
                    print("send message")
                    flag = False
                det[:, :4] = scale_coords(
                    im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls_id in det:
                    lbl = names[int(cls_id)]
                    xyxy = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
                    score = round(conf.tolist(), 3)
                    if cfg.hide_conf:
                        label = "{}".format(lbl)
                    else:
                        label = "{}: {}".format(lbl, score)
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    pred_boxes.append((x1, y1, x2, y2, lbl, score))
                    # json_data.append({"alarm": {"label":lbl,"x1":x1,"y1":y1,"x2":x2,"y2":y2}})
                    # img_json["data"].append({"alarm": {"label":lbl,"x1":x1,"y1":y1,"x2":x2,"y2":y2}})
                    # if view_img:
                    #     darknet.plot_one_box(xyxy, im0, color=(255, 0, 0), label=label)
                    annotator.box_label(xyxy, label, color=colors(c, True))
            else:
                count += 1
                if count>3000:
                    flag = True
        # t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        # print(1/sum(dt),"fps")
            # Print time (inference + NMS)

        im0 = annotator.result()
        # box_imge = imageToStr(im0)
        # img_json["box_imge"] = box_imge
        img_json["data"] = json_data
        # return json.dumps(img_json, ensure_ascii=False)
        # print(img_json)
        # break
        # cv2.imshow(str(p), im0)
        # cv2.waitKey(10)
        # print(pred_boxes)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')
        # if feed_type_curr == feed_type:
        if True:
            frame = cv2.imencode('.jpg', im0)[1].tobytes()
            time.sleep(0.02)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            # yield(json.dumps(img_json, ensure_ascii=False))
    # gc.collect()
    #         return json.dumps(img_json, ensure_ascii=False)

@server.route('/test',methods=['get', 'post'])
def video():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(detect_gen(dataset=dataset, feed_type='Camera_0'),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    # elif feed_type == 'Camera_1':
    #     return Response(detect_gen(dataset=dataset, feed_type=feed_type),
    #                     mimetype='multipart/x-mixed-replace; boundary=frame')

@server.route('/video_feed/<feed_type>',methods=['get', 'post'])
def video_feed(feed_type):
    """Video streaming route. Put this in the src attribute of an img tag."""
    if feed_type == 'Camera_0':
        return Response(detect_gen(dataset=dataset, feed_type=feed_type),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    # elif feed_type == 'Camera_1':
    #     return Response(detect_gen(dataset=dataset, feed_type=feed_type),
    #                     mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # server = pywsgi.WSGIServer(('0.0.0.0', 5000), server)
    # server.serve_forever()
    server.run(debug=True,host='0.0.0.0', port="5000", threaded=True)

