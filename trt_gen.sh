batch=1
datatype=fp16
# 1 for dynamic, 0 for static
dynamicshape=0 
rawonnx=./model/yolov3_416_batch${batch}.onnx
finalonnx=./model/yolov3_416_batch${batch}_sim.onnx
trtmodel=./model/yolov3_416_batch${batch}_${datatype}.trt
echo $rawonnx
echo $finalonnx
echo $trtmodel
if [[ -f "$rawonnx" ]]; then
  echo "$rawonnx exists."
else
  python jtest.py ${batch} ${rawonnx} ${dynamicshape}
fi
if [[ -f "$finalonnx" ]]; then
  echo "$finalonnx exists."
else
  python jtrt.py ${rawonnx} ${finalonnx} ${dynamicshape}
fi 
#python jtrt_test_int8.py ${finalonnx} ${trtmodel} ${batch} ${datatype}
python jtrt_generate.py ${finalonnx} ${trtmodel} ${batch} ${datatype}
