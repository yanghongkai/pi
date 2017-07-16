#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
print(sys.getdefaultencoding())


import numpy as np
from sklearn import metrics

import tensorflow as tf
from tensorflow.contrib import learn
import os
import yfile
import codecs
import time

FLAGS=tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_data_path',"/Users/tech/code/kcws/train.txt",'Training data dir')
tf.app.flags.DEFINE_string('valid_data_path',"/Users/tech/code/kcws/train.txt",'valid data dir')
tf.app.flags.DEFINE_string('test_data_path',"./test.txt",'Test data dir')
tf.app.flags.DEFINE_string('log_data',"log_data",'log data dir')
tf.app.flags.DEFINE_string('log_dir',"logs",'The log dir')
tf.app.flags.DEFINE_string('word2vec_path',"./vec.txt",'the word2vec data path')
tf.app.flags.DEFINE_integer('max_sentence_len',80,'max num of tokens per query')
tf.app.flags.DEFINE_integer('embedding_size',50,"embedding size")
tf.app.flags.DEFINE_integer('num_tags',2,'01')
tf.app.flags.DEFINE_integer('num_hidden',100,'hidden unit number')
tf.app.flags.DEFINE_integer('batch_size',100,'num example per mini batch')
tf.app.flags.DEFINE_integer('train_steps',50000,'training steps')
tf.app.flags.DEFINE_float("learning_rate_base",0.05,'learning rate base')
tf.app.flags.DEFINE_float("learning_rate_decay",0.99,'learning rate decay')


def do_load_data(path):
    x=[]
    y=[]
    fp=open(path,"r")
    for line in fp.readlines():
        line=line.rstrip()
        if not line:
            continue
        ss=line.split(" ")
        assert(len(ss)==(FLAGS.max_sentence_len*2))
        lx=[]
        ly=[]
        for i in xrange(FLAGS.max_sentence_len):
            lx.append(int(ss[i]))
            ly.append(int(ss[i+FLAGS.max_sentence_len]))
        x.append(lx)
        y.append(ly)
    fp.close()
    return np.array(x),np.array(y)



class Model:
    def __init__(self,embeddingSize,distinctTagNum,c2vPath,numHidden,trainMode=True,reuse=False):
        self.embeddingSize=embeddingSize
        self.distinctTagNum=distinctTagNum
        self.reuse=reuse
        self.numHidden=numHidden
        self.c2v=self.load_w2v(c2vPath)
        print(len(self.c2v))
        # len(self.c2v)=13544 file:vectrain_50.txt 13542 50
        # c2v[0]=[0,0,0,0...] 表示未填充的idx  c2v[lenghth+1]存储平均，用于表示unk
        #print(self.c2v[0])
        #print(self.c2v[-1])
        self.words=tf.Variable(self.c2v,name="words")
        self.regularizer=None
        self.global_step=tf.Variable(0,trainable=False)
        self.learning_rate=tf.train.exponential_decay(FLAGS.learning_rate_base,self.global_step,1000,FLAGS.learning_rate_decay, staircase=True)
        if trainMode:
            self.regularizer=tf.contrib.layers.l2_regularizer(0.001)
        self.trains_params=None
        self.inpx=tf.placeholder(tf.int32,shape=[None,FLAGS.max_sentence_len],name="input_placeholder")
        pass

    def get_weight_variable(self,shape,regularizer):
        weights=tf.get_variable("weights",shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(weights))
        return weights

    def length(self,data):
        used=tf.sign(tf.reduce_max(tf.abs(data),reduction_indices=2))
        # data shape [100,80,50] reduce_max shape [100,80] used shape [100,80]
        length=tf.reduce_sum(used,reduction_indices=1)
        # length shape [100]
        length=tf.cast(length,tf.int32)
        return length

    def inference(self,X,reuse=None,trainMode=True):
        word_vectors=tf.nn.embedding_lookup(self.words,X)
        # word_vectors shape [100,80,50]
        length=self.length(word_vectors)
        # length shape [100]
        length_64=tf.cast(length,tf.int64)
        if trainMode:
            word_vectors=tf.nn.dropout(word_vectors,0.5)
        with tf.variable_scope("rnn_fwbw",reuse=reuse) as scope:
            forward_output,_=tf.nn.dynamic_rnn(
                tf.nn.rnn_cell.LSTMCell(self.numHidden),
                word_vectors,dtype=tf.float32,sequence_length=length,scope="RNN_forward")
            # forward_output shape [batch_size,time_steps,num_hidden] [100,80,100]
            backward_output_,_=tf.nn.dynamic_rnn(
                tf.nn.rnn_cell.LSTMCell(self.numHidden),
                inputs=tf.reverse_sequence(word_vectors,length_64,seq_dim=1),
                dtype=tf.float32,sequence_length=length,scope="RNN_backward")
            # backward_output_ shape [100,80,100]
            backward_output=tf.reverse_sequence(backward_output_,length_64,seq_dim=1)
            output=tf.concat(2,[forward_output,backward_output])
            # output shape [100,80,200]
            output=tf.reshape(output,[-1,self.numHidden*2])
            # output [100*80,200]
            if trainMode:
                output=tf.nn.dropout(output,0.5)
        with tf.variable_scope("softmax",reuse=reuse):
            w=self.get_weight_variable([self.numHidden*2,self.distinctTagNum],self.regularizer)
            b=tf.Variable(tf.zeros([self.distinctTagNum],name='bias'))
            logits=tf.matmul(output,w)+b
            # logits shape [100*80,4]
            return logits, length

    def loss(self,X,Y):
        logits,_=self.inference(X,self.reuse,trainMode=True)
        # logits shape [100*80,4]
        logits_=tf.reshape(logits,[-1, FLAGS.max_sentence_len, self.distinctTagNum])
        # logits_ shape [100,80,4]
        cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits_,Y)
        # cross_entropy shape [100,80]
        cross_entropy_sum=tf.reduce_sum(cross_entropy,1)
        # cross_entropy_sum shape [100]
        #self.entropy=cross_entropy
        cross_entropy_mean=tf.reduce_mean(cross_entropy_sum)
        loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
        self.loss=loss
        self.train_op=tf.train.AdamOptimizer(self.learning_rate).minimize(loss,global_step=self.global_step)

    def accuracy(self):
        logits, sequence_length=self.inference(self.inpx, reuse=True, trainMode=False)
        #logits shape [100*80,4]
        logits_=tf.argmax(logits,1)
        logits_reshape=tf.reshape(logits_, [-1, FLAGS.max_sentence_len])
        self.test_inference=logits_reshape
        self.test_length=sequence_length
        infer_logits=tf.reshape(logits,[-1,FLAGS.max_sentence_len,self.distinctTagNum])
        infer_head=tf.nn.softmax(infer_logits,dim=-1,name="infer_head")


    def test_unary_score(self):
        logits=self.inference(self.inp,reuse=True,trainMode=False)
        logits_=tf.argmax(logits,1)
        logits_reshape=tf.reshape(logits_, [-1, FLAGS.max_sentence_len])
        # logits_reshape shape [100,80]
        return logits_reshape


    def load_w2v(self,path):
        fp=open(path,"r")
        print("load data from:",path)
        line=fp.readline().strip()
        ss=line.split(" ")
        total=int(ss[0])
        dim=int(ss[1])
        assert(dim==(FLAGS.embedding_size))
        ws=[]
        mv=[0 for i in range(dim)]
        # The first for 0
        ws.append([0 for i in range(dim)])
        for t in range(total):
            line=fp.readline().strip()
            ss=line.split(" ")
            assert(len(ss)==(dim+1))
            vals=[]
            for i in range(1,dim+1):
                fv=float(ss[i])
                mv[i-1]+=fv
                vals.append(fv)
            ws.append(vals)
        for i in range(dim):
            mv[i]=mv[i]/total
        ws.append(mv)
        fp.close()
        return np.asarray(ws,dtype=np.float32)


def read_csv(batch_size,file_name):
    filename_queue=tf.train.string_input_producer(file_name)
    #print(filename_queue)
    # FIFQQueue
    reader=tf.TextLineReader(skip_header_lines=0)
    key,value=reader.read(filename_queue)
    #print('-----------key,value---------')
    #print(key)
    #print(value)
    # key: Tensor("ReaderRead:0", shape=(), dtype=string)
    # value: Tensor("ReaderRead:1", shape=(), dtype=string)
    # decode_csv will convert a Tensor from type string (the text line) in a tuple of tensor columns with the specified defaults, which also sets the data type for each column
    decoded=tf.decode_csv(value,field_delim=' ',record_defaults=[[0] for i in range(FLAGS.max_sentence_len*2)])
    #print(len(decoded))
    #print(decoded)
    # len(decoded)=160 
    # decoded: [<tf.Tensor 'DecodeCSV:0' shape=() dtype=int32>, <tf.Tensor 'DecodeCSV:1' shape=() dtype=int32>, ... ]
    # batch actually reads the file and loads "batch_size" rows in a single tensor
    return tf.train.shuffle_batch(decoded,batch_size=batch_size,capacity=batch_size*50,min_after_dequeue=batch_size)




def inputs(path):
    whole=read_csv(FLAGS.batch_size,path)
    #print(whole) # 160
    #print(whole.get_shape())
    # whole: [<tf.Tensor 'shuffle_batch:0' shape=(100,) dtype=int32>,
    features=tf.transpose(tf.pack(whole[0:FLAGS.max_sentence_len]))
    label=tf.transpose(tf.pack(whole[FLAGS.max_sentence_len:]))
    return features,label


def test_evaluate(sess, inference_op, length_op, inpx, tX, tY, desc, out,step, learning_rate):
    start=time.time()
    totalEqual=0
    batchSize=FLAGS.batch_size
    totalLen=tX.shape[0]
    # totalLen 18313 tX shape [18313,80]
    numBatch=int((tX.shape[0]-1)/batchSize)+1
    correct_labels=0
    correct_lines_1=0
    correct_lines_2=0
    total_labels=0
    for i in range(numBatch):
        endOff=(i+1)*batchSize
        if endOff>totalLen:
            endOff=totalLen
        y=tY[i*batchSize:endOff]
        # y shape [100,80]
        test_logits,test_sequence_lengths= sess.run([inference_op,length_op],feed_dict={inpx: tX[i*batchSize:endOff]})
        # test_logits shape [100,80] test_sequence_length shape [100] y=shape [100,80]
        for logit, y_, sequence_length in zip(test_logits, y, test_sequence_lengths):
            # logit=shape [80] y_=shape [80] sequence_length=shape []
            logit=logit[:sequence_length]
            y_=y_[:sequence_length]
            correct=np.sum(np.equal(logit,y_))
            correct_labels+=correct
            total_labels+=sequence_length
            
            # 只判断head
            logit_res=np.where(logit==15)
            y_res=np.where(y_==15)
            if len(y_res[0])==1:
                if y_res[0][0] in logit_res[0] and len(logit_res[0])==1:
                    correct_lines_1+=1

            if len(y_res[0])==1:
                if y_res[0][0] in logit_res[0] and len(logit_res[0])<=2:
                    correct_lines_2+=1


    accuracy=100.0*correct_labels/float(total_labels)
    accuracy_head1=100.0*correct_lines_1/float(totalLen)
    accuracy_head2=100.0*correct_lines_2/float(totalLen)
    end=time.time()
    span=end-start
    print("[%s] Total:%d, Correct:%d, Accuracy:%.3f%%, Time:%.1f" % (desc,total_labels,correct_labels,accuracy,span))
    print("[%s] TotalLine:%d, Correct_lines_1:%d, Accuracy_head1:%.3f%%, Correct_lines_2:%d, Accuracy_head2:%.3f%%, Time:%.1f" % (desc,totalLen,correct_lines_1,accuracy_head1,correct_lines_2,accuracy_head2,span))
    out.write("%d\t%.4f\t%.3f\t%.3f\t%.3f\t%.1f\n" % (step,learning_rate,accuracy,accuracy_head1,accuracy_head2,span))



def main(unused_argv):
    curdir=os.path.dirname(os.path.realpath(__file__))
    trainDataPath=tf.app.flags.FLAGS.train_data_path
    filenames=yfile.getFileDir(trainDataPath)
    print(filenames)
    graph=tf.Graph()
    with graph.as_default():
        model=Model(FLAGS.embedding_size,FLAGS.num_tags,FLAGS.word2vec_path,FLAGS.num_hidden,trainMode=True,reuse=False)
        #mvalid=Model(FLAGS.embedding_size,FLAGS.num_tags,FLAGS.word2vec_path,FLAGS.num_hidden,trainMode=False,reuse=True)
    
        X,Y=inputs(filenames)
        #print(X)
        #print(Y)
        # X shape [100,80] Y shape[100,80]
        tX,tY=do_load_data(FLAGS.test_data_path)
        #print(tX)
        #print(tX.shape)
        # tX shape [18313,80]
        vX,vY=do_load_data(FLAGS.valid_data_path)
        strtime=time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime(time.time()))
        test_out_name="test-"+strtime+"-"+str(FLAGS.learning_rate_base)
        valid_out_name="valid-"+strtime+"-"+str(FLAGS.learning_rate_base)
        test_out=codecs.open(FLAGS.log_data+"/"+test_out_name, "w+",encoding="utf-8")
        valid_out=codecs.open(FLAGS.log_data+"/"+valid_out_name, "w+",encoding="utf-8")
        model.loss(X,Y)
        model.accuracy()
        #mvalid.loss()
        #mvalid.accuracy()
        sv=tf.train.Supervisor(graph=graph,logdir=FLAGS.log_dir)
        with sv.managed_session(master='') as sess:
            training_steps=FLAGS.train_steps
            #training_steps=1
            for op in tf.get_default_graph().get_operations():
                print(op.name)
            for step in xrange(training_steps):
                if sv.should_stop():
                    break
                try:

                    _, loss_val, learning_rate=sess.run([model.train_op,model.loss,model.learning_rate ])
                    if step % 100==0:
                        print("[%d] loss:[%r], learning_rate:%.4f" % (step,loss_val,learning_rate))
                    if step % 1000==0:
                        test_evaluate(sess,model.test_inference, model.test_length, model.inpx,vX,vY,"VALID",valid_out,step,learning_rate)
                        test_evaluate(sess,model.test_inference, model.test_length, model.inpx,tX,tY,"TEST",test_out,step,learning_rate)
                except KeyboardInterrupt, e:
                    valid_out.close()
                    test_out.close()
                    sv.saver.save(sess,FLAGS.log_dir+'/model',global_step=step+1)
                    raise e
            sv.saver.save(sess,FLAGS.log_dir+'/finnal-model')
            valid_out.close()
            test_out.close()
            sess.close()


if __name__=='__main__':
    tf.app.run()







