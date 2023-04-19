import json
import torch
import sys
import numpy as np
import cv2
from pathlib import Path

#from ensemble_boxes import weighted_boxes_fusion

from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.augmentations import letterbox

device = torch.device("cuda:0")
model_pb_path="/project/train/models/train/exp11/weights/best.pt"  # 模型地址一定要和测试阶段选择的模型地址一致！！！
@torch.no_grad()
def init():
    if not os.path.isfile(model_pb_path):
        
        log.error(f'{model_pb_path} does not exist')

        return None

    log.info('Loading model...')

    detection_graph = tf.Graph()

    with detection_graph.as_default():
    
        od_graph_def = tf.GraphDef()

    with tf.gfile.GFile(model_pb_path, 'rb') as fid:
    
        serialized_graph = fid.read()

    od_graph_def.ParseFromString(serialized_graph)

    tf.import_graph_def(od_graph_def, name='')

 

    log.info('Initializing session...')

    global sess

    sess = tf.Session(graph=detection_graph)

    return detection_graph


def process_image(net, input_image, args=None ):
                                                                                                               
    if not net or input_image is None:
        
        log.error('Invalid input args')

        return None

    ih, iw, _ = input_image.shape

    show_image = input_image

    if ih != input_h or iw != input_w:
    
        input_image = cv2.resize(input_image, (input_w, input_h))

 

 

input_image = np.expand_dims(input_image, axis=0)                                                                                                       
                                                                                                               




    iou_thres = 0.05  # NMS IOU threshold

    max_det = 1000  # maximum detections per image
    imgsz = [1024,1024]
    names = {
        0: 'table',
        1: 'chair',
        2: 'table_chair',
        3: 'umbrella',
        4: 'uncertain'
        }

    stride = 32
    fake_result = {}
    fake_result["model_data"] = {"objects": []}

    img = letterbox(input_image, imgsz, stride, True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img  = img/255  # 0 - 255 to 0.0 - 1.0
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, dim=0).to(device)
    img = img.type(torch.cuda.HalfTensor)
    pred = handle(img, augment=False, visualize=False)[0]

    pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)
    
    detect_objs = []
    for k, score in enumerate(pred):  # per image
        score[:, :4] = scale_coords(img.shape[2:], score[:, :4], input_image.shape).round()

        for *xyxy, conf, classes in reversed(score):
            xyxy_list = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
            conf_list = conf.tolist()
            label = np.int(classes[k])
            if label not in label_id_map:
        
                log.warning(f'{label} does not in {label_id_map}')

            continue
    
            ymin, xmin, ymax, xmax = boxes[k]

            detect_objs.append({

                'name': label_id_map[label],

                "confidence":float(score),

                'xmin': int(xmin),

                'ymin': int(ymin),

                'xmax': int(xmax),

                'ymax': int(ymax)

})

 

    result = {}

    result['algorithm_data'] = {

    "is_alert": True if len(detect_objs) > 0 else False,

    "target_count": len(detect_objs),

    "target_info": detect_objs

}

    result['model_data'] = {"objects": detect_objs}

    return json.dumps(result, indent = 4)


if __name__ == '__main__':
    # Test API
    img =cv2.imread('/home/data/878/dining_out_public_roads_avenue_multiplex_train_p_day_202020304_10473.jpg')
    predictor = init()
    import time
    s = time.time()
    fake_result = process_image(predictor, img)
    e = time.time()
    print(fake_result)
    print((e-s))