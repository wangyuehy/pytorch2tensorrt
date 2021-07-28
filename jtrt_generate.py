from __future__ import print_function

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import pdb
import numpy as np
import sys, os
TRT_LOGGER = trt.Logger()
#TRT_LOGGER.min_severity = trt.Logger.Severity.VERBOSE
TRT_LOGGER.min_severity = trt.Logger.Severity.INFO
from itertools import chain
import argparse
import os


try:
    # Sometimes python does not understand FileNotFoundError
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


######### prepare calibrate data #############

def cv2_preprocess(img,target_size=(416,416),mean=[0,0,0],std=[255.,255.,255.]):
  h,w,c = img.shape
  #target_size = 416
  #mean = [0,0,0]
  #std=[255.,255.,255.]
  mean = np.float64(np.array(mean).reshape(1,-1))
  to_rgb=True
  stdinv = 1/np.float64(np.array(std).reshape(1,-1))

  # resize
  scale_h = target_size[0]/h
  scale_w = target_size[1]/w
  if target_size[0]/ h > target_size[1]/w:
    scale = target_size[1]/w
  else:
    scale = target_size[0]/h
  dim = (int(w*scale),int(h*scale))  # notice w,hw sequence
  res_img = cv2.resize(img, dim,interpolation=cv2.INTER_LINEAR)
  resized_img = res_img.copy().astype(np.float32)

  # norm
  if to_rgb:
    cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB, resized_img)
  cv2.subtract(resized_img, mean, resized_img)
  cv2.multiply(resized_img, stdinv, resized_img)

  # pad
  padding = (0,0, target_size[1] - resized_img.shape[1], target_size[0] - resized_img.shape[0])
  pad_img = cv2.copyMakeBorder(resized_img, padding[1],padding[3],padding[0],padding[2],
                        cv2.BORDER_CONSTANT,value=0)

  # transpose
  img = np.ascontiguousarray(pad_img.transpose(2,0,1))

  return img
  
def imgs_preprocess(imgs,target_size=(416,416)):
  #target_size = 416
  mean = [0,0,0]
  std=[255.,255.,255.]

  datas = np.random.random((len(imgs),3,target_size[0],target_size[1]))
  for ith,img in enumerate(imgs):
    fimg = cv2.imread(img)
    preprocessed_img = cv2_preprocess(fimg,target_size=target_size,mean=mean,std=std)
    datas[ith,:]=preprocessed_img
  return datas

def load_COCO_data(num_cali=100, target_size=(416,416)):
    coco_path = '/workspace/public/dataset/cv/coco/val2017'
    imgindex = np.random.randint(0,5000,num_cali)
    #imglist = os.listdir(coco_path)[:num_cali]
    imglist = os.listdir(coco_path)
    imglist = [os.path.join(coco_path,imglist[inx]) for inx in imgindex]
    datas = imgs_preprocess(imglist,target_size)
    return datas


class CoCoEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file, batch_size=8, input_shape=(416,416)):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file
  
        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.data = load_COCO_data(8, input_shape)
        self.batch_size = batch_size
        self.current_index = 0

        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

        batch = self.data[self.current_index:self.current_index + self.batch_size].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            print('read cache ',self.cache_file)
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        print('write cache ',cache)
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def mark_outputs(network):
    # Mark last layer's outputs if not already marked
    # NOTE: This may not be correct in all cases
    last_layer = network.get_layer(network.num_layers-1)
    if not last_layer.num_outputs:
        print("Last layer contains no outputs.")
        return

    for i in range(last_layer.num_outputs):
        network.mark_output(last_layer.get_output(i))

def check_network(network):
    if not network.num_outputs:
        print("No output nodes found, marking last layer's outputs as network outputs. Correct this if wrong.")
        mark_outputs(network)
    
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    max_len = max([len(inp.name) for inp in inputs] + [len(out.name) for out in outputs])

    print("=== Network Description ===")
    for i, inp in enumerate(inputs):
        print("Input  {0} | Name: {1:{2}} | Shape: {3}".format(i, inp.name, max_len, inp.shape))
    for i, out in enumerate(outputs):
        print("Output {0} | Name: {1:{2}} | Shape: {3}".format(i, out.name, max_len, out.shape))

def add_profiles(config, inputs, opt_profiles):

    for i, profile in enumerate(opt_profiles):
        for inp in inputs:
            _min, _opt, _max = profile.get_shape(inp.name)
        config.add_optimization_profile(profile)

# TODO: This only covers dynamic shape for batch size, not dynamic shape for other dimensions
def create_optimization_profiles(builder, inputs, batch_sizes=[1,8,16,32,64]): 
    # Check if all inputs are fixed explicit batch to create a single profile and avoid duplicates
    if all([inp.shape[0] > -1 for inp in inputs]):
        profile = builder.create_optimization_profile()
        for inp in inputs:
            fbs, shape = inp.shape[0], inp.shape[1:]
            print("fbs:{}".format(fbs))
            print("shape:{}".format(shape))
            profile.set_shape(inp.name, min=(fbs, *shape), opt=(fbs, *shape), max=(fbs, *shape))
            return [profile]
    
    # Otherwise for mixed fixed+dynamic explicit batch inputs, create several profiles
    profiles = {}
    for bs in batch_sizes:
        if not profiles.get(bs):
            profiles[bs] = builder.create_optimization_profile()

        for inp in inputs: 
            shape = inp.shape[1:]
            # Check if fixed explicit batch
            if inp.shape[0] > -1:
                bs = inp.shape[0]
            profiles[bs].set_shape(inp.name, min=(bs, *shape), opt=(bs, *shape), max=(bs, *shape))
    return list(profiles.values())



def gen_engine(onnx_file_path, engine_file_path="", batch_size=8,HW=(416,416),datatype=1):  #datatype 2 for fp16, 1 for int8, 0 for fp32
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder,\
            builder.create_network(EXPLICIT_BATCH) as network, \
                builder.create_builder_config() as config, \
                    trt.OnnxParser(network, TRT_LOGGER) as parser:
            print('building batchsize {} input shape {}'.format(batch_size,HW))
            #batch_size = global_batch_size
            builder.max_batch_size = batch_size
            #config.max_workspace_size = 1 << 33 # 8GB
            #max_workspace_size = 1 << 30
            max_workspace_size = 1 <<31
            config.max_workspace_size = max_workspace_size
            with open(onnx_file_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            print('Completed parsing of ONNX file')
            check_network(network)    # not neccessory

            # int8
            if datatype == 1:   
                config.set_flag(trt.BuilderFlag.INT8)
                calibration_cache = "coco_calibration.cache"
                #calibration_cache="/workspace/coco_calibration.cache"
                #if os.path.exists(calibration_cache):
                #    os.remove(calibration_cache)
                #    print('remove ',calibration_cache)
                calib = CoCoEntropyCalibrator(cache_file=calibration_cache, batch_size=batch_size,input_shape=HW)
                config.int8_calibrator = calib
            elif datatype ==2:
                config.set_flag(trt.BuilderFlag.FP16)
            else:
                pass
            batch_sizes = [batch_size]
            inputs = [network.get_input(i) for i in range(network.num_inputs)]
            opt_profiles = create_optimization_profiles(builder, inputs, batch_sizes)
            add_profiles(config, inputs, opt_profiles)
            #config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_engine(network, config)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()



def main(onnx_file_path='./tmodel/yolov3_416_batch8_sim.onnx',engine_file_path='./tmodel/yolov3_416_batch8_int8.trt',batch_size=8,datatype=1,input_HW=(416,416)):
    #input_HW=(416,416)
    #datatype = 1  # 0 for fp32, 1 for int8, 2 for fp16
    #batch_size = 8
    
    if len(sys.argv) ==3:
        onnx_file_path = sys.argv[1]
        engine_file_path = sys.argv[2]
    elif len(sys.argv) == 4:
        onnx_file_path = sys.argv[1]
        engine_file_path = sys.argv[2]
        batch_size = int(sys.argv[3])
    elif len(sys.argv) == 5:
        onnx_file_path = sys.argv[1]
        engine_file_path = sys.argv[2]
        batch_size = int(sys.argv[3])
        if sys.argv[4] == 'int8':
            datatype = 1
        elif sys.argv[4] == 'fp16':
            datatype = 2
        elif sys.argv[4] =='fp32':
            datatype = 0
        else:
            print('error input datatype')
            return
        print('datatype is ',sys.argv[4])
    else:
        pass
    print('-'*30)
    print('onnx {}\nengine {}\nbatch {}\ndatatype {}\ninput size {}'.format(onnx_file_path,engine_file_path,batch_size,datatype,input_HW))
    image = np.random.random((batch_size,3,input_HW[0],input_HW[1]))
    trt_outputs = []

    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    with gen_engine(onnx_file_path,  engine_file_path=engine_file_path, batch_size=batch_size, HW=input_HW,datatype=datatype) as engine, engine.create_execution_context() as context:
        print('Start testing ...')
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        inputs[0].host = image
        trt_outputs = do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
   
    #print(trt_outputs)

def test_main():
  engine_file_path = "yolov3_test.trt"
  if os.path.exists(engine_file_path):
    print('removing ',engine_file_path)
    os.remove(engine_file_path)
  
  mode = 1
  if 1 == mode:
  # static mode
    onnxfile = 'tmodel/yolov3_416_batch8_sim.onnx'
    batch_size = 8
    #int8
    datatype =1
    input_shape=(416,416)
  elif 2 == mode:
    #fp16
    onnxfile = 'tmodel/yolov3_416_batch8_sim.onnx'
    batch_size = 8
    datatype = 2
    input_shape=(416,416)
  elif 3 ==mode:
    # dynamic mode
    # int8
    # batch 1
    onnxfile = 'tmodel/yolov3_416_dyn_sim_v1.onnx'
    batch_size =1
    datatype = 1
    input_shape=(416,416)
  elif 4 == mode:
    # int8
    # batch 8
    onnxfile = 'tmodel/yolov3_416_dyn_sim_v1.onnx'
    batch_size = 8
    datatype = 1
    input_shape=(416,416)
  elif 5 == mode:
    # fp16
    # batch 1
    onnxfile = 'tmodel/yolov3_416_dyn_sim_v1.onnx'
    batch_size =1
    datatype = 2
    input_shape=(416,416)
  elif 6 ==mode:
    # fp16
    # batch 8
    onnxfile = 'tmodel/yolov3_416_dyn_sim_v1.onnx'
    batch_size =8
    datatype = 2   
    input_shape=(416,416)
  else:
    pass
  main(onnxfile, engine_file_path,batch_size,datatype,input_shape)

if __name__ =='__main__':
  main()


