

import onnxruntime

import numpy as np

import onnx

import pdb

DEBUG = 0

def generatedata():

    image = np.load('jtestdata.npy')

    n,c,h,w = image.shape

    uppad = int((320-h)/2)

    downpad = int(320 - h - uppad)

    plate = np.pad(image[0],((0,0),(uppad,downpad),(0,0)), 'constant', constant_values=0.0)

    np.save('jpaddata.npy',[plate])



def test(onnxfile,debug=DEBUG):

    print('inference with ',onnxfile)

    ort_session = onnxruntime.InferenceSession(onnxfile)

    img = np.load('jpaddata.npy')

    ort_inputs = {ort_session.get_inputs()[0].name: img}

    ort_outputs = ort_session.run(None, ort_inputs)

    if debug:
        
        print('position\n',result[0][0][:10])
        
        print('label\n',result[1][0][:10])

    return ort_outputs

def test2():

    pdb.set_trace()

    import onnx_graphsurgeon as gs

    graph = gs.import_onnx(onnx.load('jstmp.onnx'))

    outputs = []

    #for node in graph.nodes:

    #    if node.name == 'Concat_747':

    #        outputs.append(node.outputs[0])

    #    if node.name == 'Mul_762':

    #        outputs.append(node.outputs[0])

    #graph.outputs = outputs

    #graph.cleanup()
    #nms = [node for node in graph.nodes if node.op == 'NonMaxSuppression' ][0]

    node = [node for node in graph.nodes if node.name == 'Add_449' ][0]

    graph.outputs = node.outputs
    
    node = [node for node in graph.nodes if node.name == 'TopK_435' ][0]

    graph.outputs.append(node.outputs[0])
   

    def insertnode(graph, node):
        floattensor = node.inputs[1]
        cast_int32 = gs.Variable(name='cast_int32'+node.name,dtype=np.int32)
        cast_node = gs.Node(op='Cast',inputs=[floattensor],outputs=[cast_int32],attrs={"to":onnx.TensorProto.INT32})
        node.inputs[1] = cast_int32
        graph.nodes.append(cast_node)

    graph.cleanup()
    for node in graph.nodes:
        if node.name in['Add_449','Add_587','Add_725']:
            insertnode(graph,node)  
            pass
    onnx.save(gs.export_onnx(graph),'jstmp_short.onnx')

    #result = test('jstmp_short.onnx')

    #print('position\n',result[0][0][:10])
        
    #print('label\n',result[1][0][:10])

def test3(onnxfile,nodes):
    # cut middle of graph to output

    pdb.set_trace()

    import onnx_graphsurgeon as gs

    ograph = onnx.load(onnxfile)

    graph = gs.import_onnx(onnx.load(onnxfile))

    outputs = []
    graph.outputs = []
    for nodename in nodes:
        for gnode in graph.nodes:
            if nodename == gnode.name:
                graph.outputs.append(gnode.outputs[0])
                outputs.append(nodename)
   
    def insertnode(graph, node):
        floattensor = node.inputs[1]
        cast_int32 = gs.Variable(name='cast_int32'+node.name,dtype=np.int32)
        cast_node = gs.Node(op='Cast',inputs=[floattensor],outputs=[cast_int32],attrs={"to":onnx.TensorProto.INT32})
        node.inputs[1] = cast_int32
        graph.nodes.append(cast_node)

    for node in graph.nodes:
        if node.name in['Add_449','Add_587','Add_725']:
            insertnode(graph,node)  

    graph.cleanup()

    model = gs.export_onnx(graph)

    #onnx.checker.check_model(model)

    onnx.save(model, 'jstmp_mid2.onnx')

    #result = test('jstmp_mid.onnx')

    #for name, res in zip(outputs,result):
    #    print('-'*36)
    #    print(name)
    #    print(res)
        #np.save('./bnmsplugin/'+name+'.npy',res)

def test4(onnxfile,nodes):
    '''
        output a mid nodes output
    '''
    pdb.set_trace()

    model = onnx.load(onnxfile)

    for node in model.graph.node:
        if node.name in nodes:
            for output in node.output:

                model.graph.output.extend([onnx.ValueInfoProto(name=output)])

    ort_session = onnxruntime.InferenceSession(model.SerializeToString())
    
    img = np.load('jpaddata.npy')

    ort_inputs = {ort_session.get_inputs()[0].name: img}

    outputs = [x.name for x in ort_session.get_outputs()]

    ort_outs = ort_session.run(outputs, ort_inputs)

    ort_outs = zip(outputs, ort_outs)

    for i,j in ort_outs:
        print(i)
        print(j)
        print('-'*30)

   
def test5(onnxfile,nodes):
    # cut middle of graph to output

    pdb.set_trace()

    import onnx_graphsurgeon as gs

    graph = gs.import_onnx(onnx.load(onnxfile))

    outputs = []
    graph.outputs = []
    for nodename in nodes:
        for gnode in graph.nodes:
            if nodename == gnode.name:
                graph.outputs.append(gnode.outputs[0])
                outputs.append(nodename)
    
    graph.cleanup()
    def insertnode(graph, node):
        floattensor = node.inputs[1]
        cast_int32 = gs.Variable(name='cast_int32'+node.name,dtype=np.int64)
        cast_node = gs.Node(op='Cast',inputs=[floattensor],outputs=[cast_int32],attrs={"to":onnx.TensorProto.INT64})
        node.inputs[1] = cast_int32
        graph.nodes.append(cast_node)

    for node in graph.nodes:
        if node.name in['Add_449','Add_587','Add_725']:
            #insertnode(graph,node)  
            pass
        if node.op =='TopK':
            indix= gs.Variable(name=node.name+'_out1',dtype=np.float32)
            node.outputs.insert(0,indix)

    #

    model = gs.export_onnx(graph)

    #onnx.checker.check_model(model)

    onnx.save(model, 'jstmp_mid2.onnx')

    result = test('jstmp_mid2.onnx')
    

    for name, res in zip(outputs,result):
        print('-'*36)
        print(name)
        print(res)
        #np.save('./bnmsplugin/'+name+'.npy',res)

if __name__ == '__main__':
    '''
        a. compare onnx model with pytorch model
        
        result: 
                lable and socre is correct
                onnx position is 0.5x of pytorch model, due to the scaling factor of data
                pad also influence the precision

        b. compare onnx model with simplifiered model

        result:
                data are same

        c. compare truncate trt with simplified onnx model

        result:
                
    '''

    '''
    onx = 'tmp.onnx'
    result = test(onx)
    
    onx = 'jstmp.onnx'
    result = test(onx)
    '''

    #test2()
    #test3('jstmp.onnx',['Concat_747','Mul_762'])
    #test3('jstmp.onnx',['Concat_552','Concat_414'])
    #test5('jstmp.onnx',['Concat_747','Mul_762'])
    test5('jstmp.onnx',['Concat_747','Mul_762'])
    #test4('jupdate.onnx',['Concat_747','Mul_762'])
    #test('jstmp_short.onnx')
    #print('result \n',result)
    #test4('jstmp.onnx',['NonMaxSuppression_776'])
