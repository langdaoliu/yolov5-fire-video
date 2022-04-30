from easydict import EasyDict

cfg = EasyDict()  # 访问属性的方式去使用key-value 即通过 .key获得value

# cfg.source = "streams.txt"  # 为视频图像文件地址

#cfg.source = "E:/workshop/taobao/fire/video/demo1.mov"  # 为视频图像文件地址
#cfg.source = "E:/workshop/taobao/fire/video/demo2.mov"  # 为视频图像文件地址
cfg.source = "E:/workshop/taobao/fire/video/demo3.mov"  # 为视频图像文件地址
#cfg.source = "E:/workshop/taobao/fire/video/demo4.mov"  # 为视频图像文件地址
# cfg.source = "E:/workshop/taobao/fire/fire-smoke111/fire-smoke-new/0047.jpg"
cfg.weights = 'best.engine' # 自己的模型地址
cfg.device =  "0" # 使用的device类别，如是GPU，可填"0"
cfg.imgsz = 640  # 输入图像的大小
cfg.stride = 32  # 步长
cfg.conf_thres = 0.35 # 置信值阈值
cfg.iou_thres = 0.45  # iou阈值
cfg.augment = False  # 是否使用图像增强
cfg.dnn = False      # use OpenCV DNN for ONNX inference
cfg.data = "data/fire_smoke.yaml"   # dataset.yaml path
cfg.half = False  # use FP16 half-precision inference
cfg.souce = 'data/images' # file/dir/URL/glob, 0 for webcam
cfg.imgsz = (640, 640)  # inference size (height, width)
cfg.max_det = 1000
cfg.agnostic_nms = False  # class-agnostic NMS
cfg.hide_labels = False,  # hide labels
cfg.hide_conf = True,  # hide confidences

