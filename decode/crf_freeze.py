#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
import os,argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util

dir=os.path.dirname(os.path.realpath(__file__))
print(dir)

def freeze_graph(model_folder):
    # We retrieve our checkpoint fullpath
    checkpoint=tf.train.get_checkpoint_state(model_folder)
    input_checkpoint=checkpoint.model_checkpoint_path
    #print(input_checkpoint)
    # input_checkpoint=/home/yhk/gitproject/pi/test/results/graph.chkp

    # We precise the file fullname of our freezed graph 
    absolute_model_folder="/".join(input_checkpoint.split('/')[:-1])
    #print(absolute_model_folder)
    # absolute_model_folder=/home/yhk/gitproject/pi/test/results
    output_graph=absolute_model_folder+"/frozen_model.pb"

    # Before exporting our graph, we need to precise what is our output node 
    # this variable is plural, because you can hvae multiple output nodes
    # freeze之前必须明确哪个是输出节点，也就是我们要得到推论结果的节点
    # 只有定义了输出结点，freeze才会把输出节点所必要的结点都保存下来，或者哪些节点可以丢弃
    # 所以，output_node_names必须根据不同的网络进行修改
    output_node_names="softmax_1/unary_scores,length_1,transitions"

    # We clear the devices, to allow tensorflow to control on the loading where it wants operations to be calculated
    clear_devices=True

    # We import the meta graph and retrieve a Saver
    saver=tf.train.import_meta_graph(input_checkpoint+".meta",clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph=tf.get_default_graph()
    input_graph_def=graph.as_graph_def()

    #We start a session and restore the graph weights
    #这边已经将训练好的参数加载进来，也即最后保存的模型是有图，并且图里面已经有参数了，所以才叫做时frozen
    # 相当于将参数已经固化在了图中
    with tf.Session() as sess:
        saver.restore(sess,input_checkpoint)

        # We use a built-in TF helper to export variables to constant
        output_graph_def=graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names.split(",")) # We split on comma for convenience
        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph,"wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))




if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_folder",type=str,help="Model folder to export")
    args=parser.parse_args()

    freeze_graph(args.model_folder)






















