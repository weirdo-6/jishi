import json
import torch
import sys
import numpy as np
import cv2
import os
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

 # Extract image tensor

    image_tensor = net.get_tensor_by_name('image_tensor:0')

# Extract detection boxes, scores, classes, number of detections

    boxes = net.get_tensor_by_name('detection_boxes:0')

    scores = net.get_tensor_by_name('detection_scores:0')

    classes = net.get_tensor_by_name('detection_classes:0')

    num_detections = net.get_tensor_by_name('num_detections:0')

# Actual detection.

    (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: input_image})


    scores = np.squeeze(scores)

    valid_index = len(scores[scores >= 0.5])

 

    boxes = np.squeeze(boxes)[:valid_index]

    boxes[:, 0] *= ih

    boxes[:, 2] *= ih

    boxes[:, 1] *= iw

    boxes[:, 3] *= iw

    boxes = boxes.astype(np.int32)

    classes = np.squeeze(classes)[:valid_index]

    scores = scores[:valid_index]
 
    rat_class_id = 1

    category_index = {int(rat_class_id): {'id': 1, 'name': 'rat'}}

    plot_save_detections(show_image,boxes,classes.astype(int),scores,category_index,image_name = "test_image.jpg")

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