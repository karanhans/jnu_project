POTHOLE DETECTION

We are developing a model to detect potholes on roads . We are developing this model for night mode also . 
We have used a hybrid model of Yolo-v2 and mask rcnn.

YOLO-V2

Weights - https://drive.google.com/open?id=1VwViHDdc4W8t28rMiuGjxChiAtFxMAjN

Then run this command -> python predict.py -c config.json -w /path/to/best_weights.h5 -i /path/to/image/or/video



MASK_RCNN

Weights - https://drive.google.com/file/d/1JA_xsHkohFiX-T7vSGefpTGemqMbYDCt/view

Train Dataset - https://github.com/ytdl-org/youtube-dl#installation
Test Dataset - https://drive.google.com/drive/u/0/folders/1duZ9O0If8mpHk8lZkFHQifv5R8z4dcKx

Custom training process - python3 custom.py train --dataset=customImages --weights=mask_rcnn_damage_0160.h5 --logs logs/
Background training process - python3 custom.py train --dataset=customImages --weights=mask_rcnn_damage_0160.h5 --logs logs/&



Hybrid model of Mask_Rcnn and YOLO-V2

Training Dataset - https://drive.google.com/file/d/1PGhXUnaJDpcjgoLEhNn9gRvGZGRCzWUt/view?usp=sharing
Testing Dataset - https://drive.google.com/file/d/1PGhXUnaJDpcjgoLEhNn9gRvGZGRCzWUt/view?usp=sharing

Weights - https://drive.google.com/open?id=1VwViHDdc4W8t28rMiuGjxChiAtFxMAjN