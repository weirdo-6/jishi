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