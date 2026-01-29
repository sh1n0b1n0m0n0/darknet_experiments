$DN_BIN/darknet detector \
        calc_anchors \
        /home/alexsh/temp/rv_yolo_1cls_plates/cfg/obj.data \
        -num_of_clusters 6 \
        -width 416 \
        -height 256 \
        -show