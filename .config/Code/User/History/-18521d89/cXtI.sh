#run.sh
rm  -r /project/train/src_repo/dataset
#创建数据集相关文件夹
mkdir /project/train/src_repo/dataset
mkdir /project/train/src_repo/dataset/Annotations
mkdir /project/train/src_repo/dataset/images
mkdir /project/train/src_repo/dataset/ImageSets
mkdir /project/train/src_repo/dataset/labels
mkdir /project/train/src_repo/dataset/ImageSets/Main

cp /home/data/878/*.xml /project/train/src_repo/dataset/Annotations
cp /home/data/878/*.jpg  /project/train/src_repo/dataset/images
#执行数据集划分、转换
python /project/train/src_repo/split_train_val.py --xml_path /project/train/src_repo/dataset/Annotations  --txt_path /project/train/src_repo/dataset/ImageSets/Main
cp /project/train/src_repo/voc_label.py /project/train/src_repo/dataset
python /project/train/src_repo/dataset/voc_label.py
#执行YOLOV5训练脚本
python /project/train/src_repo/yolov5/train.py   --data helmet.yaml --project /project/train/models/train  