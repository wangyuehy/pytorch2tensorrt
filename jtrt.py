import onnx_graphsurgeon as gs
#import graphsurgeon as gs
import numpy as np
import onnx
import pdb
import sys


def mainv5(inmodel='yolov3_batch8_sim.onnx', outmodel='jupdate.onnx',dynamic_shape=True, use_simplify=True):

    #dynamic_shape = True
    #use_simplify = True
    if use_simplify:
        import onnxsim
        tgraph = onnx.load(inmodel)
        if dynamic_shape:
            input_shapes={}
            input_shapes['input'] = [8,3,416,416]
            model_sim, _ = onnxsim.simplify(tgraph, dynamic_input_shape=True,input_shapes=input_shapes)
        else:
            model_sim, _ = onnxsim.simplify(tgraph)
        graph = gs.import_onnx(model_sim)
    else:
        graph = gs.import_onnx(onnx.load(inmodel))
    #graph = gs.import_onnx(onnx.load('jstmp.onnx'))
    #nms = [node for node in graph.nodes if node.op =='NonMaxSuppression'][0]
    nms,transpose=None,None
    for node in graph.nodes:
        if node.op == 'NonMaxSuppression':
            nms = node
    
    boxes = nms.inputs[0]
    #scores = nms.inputs[1]
    transpose_out = nms.inputs[1]
    for node in graph.nodes:
        if node.op =='Transpose' and transpose_out in node.outputs:
            transpose = node
    score_pretrans_tensor = transpose.inputs[0]
    #transpose.outputs.clear()

    keepTopK = 200
    #batch_size = 1
    num_detections = gs.Variable(name='num_detection',dtype=np.int32,shape=(-1,1))
    nmsed_boxes = gs.Variable(name='nmsed_boxes',dtype=np.float32,shape=(-1,keepTopK,4))
    nmsed_scores = gs.Variable(name='nmsed_scores',dtype=np.float32,shape=(-1,keepTopK))
    nmsed_classed = gs.Variable(name='nmsed_classed',dtype=np.float32,shape=(-1,keepTopK))
    
    #axes = gs.Constant(name='unsqueeze_axes',values=np.array([2]).astype(np.int64))
    unsqueeze_out = gs.Variable(name='unsqueezed_output',dtype=np.float32)
    unsqueeze_node = gs.Node(op='Unsqueeze',inputs=[boxes],outputs=[unsqueeze_out],attrs={"axes":[2]})
    graph.nodes.append(unsqueeze_node)
    
    out1 = gs.Variable(name='out1',dtype=np.float32)
    #batchnms = gs.Node(op='BatchedNMSDynamic_TRT',inputs=[unsqueeze_out,scores],
    if dynamic_shape:
        batchnms = gs.Node(op='BatchedNMSDynamic_TRT',inputs=[unsqueeze_out,score_pretrans_tensor],
            outputs=[num_detections,nmsed_boxes,nmsed_scores,nmsed_classed],
            attrs={"shareLocation":True,
            "backgroundLabelId":-1,
            'numClasses':80,
            'topK':1000,
            'keepTopK':keepTopK,
            'scoreThreshold':0.1,
            'iouThreshold':0.45,
            'isNormalized':False,
            'clipBoxes':False,
            'scoreBits':16}
            )
    else:
        batchnms = gs.Node(op='BatchedNMS_TRT',inputs=[unsqueeze_out,score_pretrans_tensor],
            outputs=[num_detections,nmsed_boxes,nmsed_scores,nmsed_classed],
            attrs={"shareLocation":True,
            "backgroundLabelId":-1,
            'numClasses':80,
            'topK':1000,
            'keepTopK':keepTopK,
            'scoreThreshold':0.1,
            'iouThreshold':0.45,
            'isNormalized':False,
            'clipBoxes':False,
            'scoreBits':16}
            )
    graph.nodes.append(batchnms)
    graph.outputs = [num_detections,nmsed_boxes,nmsed_scores,nmsed_classed]
    graph.cleanup()
    for node in graph.nodes:
        if node.op =='TopK':
            indix= gs.Variable(name=node.name+'_out1',dtype=np.float32)
            node.outputs.insert(0,indix)

    onnx.save(gs.export_onnx(graph), outmodel)

if __name__ == '__main__':
    DEBUG = False
    RUN = not DEBUG
    dynamic_shape = False
    use_simplify = True
    if DEBUG:
        #mtest3()
        mtest4()
    elif RUN:
        if len(sys.argv) ==4:
            inmodel = sys.argv[1]
            outmodel = sys.argv[2]
            dynamic_shape_flag = int(sys.argv[3])
            if dynamic_shape_flag == 1:
                dynamic_shape = True
            else:
                dynamic_shape = False
            mainv5(inmodel,outmodel,dynamic_shape=dynamic_shape,use_simplify=use_simplify)
        else:
            mainv5(dynamic_shape=dynamic_shape,use_simplify=use_simplify)
    else:
        pass
