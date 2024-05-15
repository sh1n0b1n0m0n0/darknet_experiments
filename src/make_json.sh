/home/alexsh/roadview_research/code/darknet/darknet detector test \
    -dont_show \
    data_lpr/prepared/obj.data \
    yolov4-tiny_17cls_250424_front/yolov4-tiny_17cls.cfg \
    yolov4-tiny_17cls_250424_front/train/backup/yolo_best.weights \
    -ext_output \
    -out result_is_front.json < /home/alexsh/darknet_experiments/data_lpr/prepared/test_israel_front.txt \
    -thresh 0.25