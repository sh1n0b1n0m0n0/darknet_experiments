/home/alexsh/roadview_research/code/darknet/darknet detector test \
    -dont_show \
    /home/alexsh/darknet_experiments/models_lpr/yolov4_17cls_170524_rear-front_640x640/obj.data \
    /home/alexsh/darknet_experiments/models_lpr/yolov4_17cls_170524_rear-front_640x640/yolov4_17cls.cfg \
    /home/alexsh/darknet_experiments/models_lpr/yolov4_17cls_170524_rear-front_640x640/train/backup/yolov4_17cls_best.weights \
    -ext_output \
    -out /home/alexsh/darknet_experiments/models_lpr/yolov4_17cls_170524_rear-front_640x640/eval/results.json < /home/alexsh/darknet_experiments/data_lpr/prepared/test_17cls_ptl_filtered.txt \
    -thresh 0.25