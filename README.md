# YOLO-v2-Pytorch 

hello guys, this is a yol0-v2 implementation base pytorch 

if you think it's helpful for you, plz give me a star

I will so appreciate it!

## Custom dataset
1. transfer your dataset to Pascal-VOC format
2. modify the  dataset root dirctory in voc0712.py line:28 
3. plz run the kmeans for your dataset,this version of yolo is is anchor based.so,you have to do this type shit
## Backbone
- ResNet-50

## Neck
- Reorg layer
- 

## Detection head
- Convs

## Loss function
- conf_loss_function = MSEWithLogitsLoss
- cls_loss_function = CrossEntropyLoss
- offset_xy_loss_function = BCEWithLogitsLoss
- w_h_loss_function = MSELoss

## Post process 
- NMS

## experiment
-  Pascal VOC
