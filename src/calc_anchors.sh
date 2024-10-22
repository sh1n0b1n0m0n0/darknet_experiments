$DN_BIN/darknet detector \
        calc_anchors \
        /home/alexsh/darknet_experiments/data_lpr_3cls/prepared/obj.data \
        -num_of_clusters 6 \
        -width 416 \
        -height 256 \
        -show