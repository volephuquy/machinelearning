# Object Detection （物体検出）
- 2012年開催された大規模画像認識のコンペILSVRC(ImageNet Large Scale Visual Recognition Challenge) で[AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)が圧倒的な成績で優勝
- 一般的にコンピュータ・ビジョンのカテゴリは以下のように大きく3つに分類されます。
  - Classification : 各画像ごとにラベルの分類
  - Object Detection : 画像内で検出された各物体領域ごとにラベルの分類
  - Segmentation : 画像内の各pixelごとにラベルの分類 (Semantic or Instance)
- 物体検出の分野で重要とされている指標は、mAP (mean Average Precision) と FPS (Frame Per Second)
```
20 classes:
Person: person
Animal: bird, cat, cow, dog, horse, sheep
Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train
Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor
Train/validation/test: 9,963 images containing 24,640 annotated objects
```

- MS COCO (Microsoft Common Object in Context) data set: 80分類のdata set
```
person, bicycle, car, motorbike, aeroplane, bus, train, truck, boat, traffic light,
fire hydrant, stop sign, parking meter, bench, 
bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite
baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass,
cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot
hot dog, pizza, donut, cake, chair, sofa, pottedplant, bed, diningtable, toilet, tvmonitor
laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator
book, clock, vase, scissors, teddy bear, hair drier, toothbrush
```

- TensorFlowなどのライブラリに実装されている物体検出のアルゴリズムは
  - Region Proposal : 物体らしい箇所を検出
  - Classification : 検出箇所のラベル分類
  - Bounding Box Regression : 正確な物体を囲う位置座標を推定
  
- 物体検出（Object Detection）は、ある画像の中から特定された物体の位置とカテゴリー(クラス)を検出する
- Deep Learning(CNN)で一般物体検出アルゴリズムで有名な３種類：
  - R-CNN (Regions with CNN features)の発展系Faster R-CNN
  - YOLO(You Only Look Once)
  - SSD: Single Shot MultiBox Detector

# やってみる
- [TensorFlow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
- intall prepare
  - `conda install -c anaconda cython`
  - `conda install -c conda-forge pycocotools`
  - brew install protobuf
