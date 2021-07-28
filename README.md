# pytorch2tensorrt
this is an example of python implementation for PyTorch to tensorrt conversion.

example model is yolov3 416x416 from mmdetection.

script function
  jtest.py:             Generate onnx from mmdetection pytorch model
  jtrt.py:              Add batchedNMS plugin to onnx model
  jtrt_generate.py:     Generate fp16/int8/fp32 model
  onnx_change_batch.py: Change the batch size of a fixed batchsize onnx model
  jonnxruntime.py:      An example script to insert/delete node in onnx with onnx_graphsurgeon
  trt_gen.sh:           An example test script

a general pipline
  1. generate raw onnx model from mmdetection
    python jtest.py $batch $rawonnx $dynamicshape
  2. since op "NMS" in onnx model cannot parse by tensorrt, so use graphsurgeon to delete "NMS" op 
  and insert a "BatchNMS_TRT" op. and do simplify to remove unused op.
    python jtrt.py $rawonnx $finalonnx $dynamicshape
  3. generate trt model 
    python jtrt_generate.py $finalonnx $trtmodel $batch $datatype
tips:
  a. $dynamicshape could be [ 0 , 1 ], 0 for static model, 1 for dynamic model
  b. $datatype could be [ int8 , fp16 , fp32 ]
  c. onnx_change_batch.py is used in static mode, where you could only generate X batch model with 1 raw onnx model
  
