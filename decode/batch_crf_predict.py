#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
import tensorflow as tf
import argparse
import numpy as np
import codecs

#LABELS={'HED':15, 'SBV':1, 'VOB':2, 'IOB':3, 'FOB':4, 'DBL':5, 'ATT':6, 'ADV':7, 'CMP':8, 'COO':9, 'POB':10, 'LAD':11, 'RAD':12, 'IS':13, 'WP':14}
LABELS=['PAD','SBV','VOB','IOB','FOB','DBL','ATT','ADV','CMP','COO','POB','LAD','RAD','IS','WP','HED']
def initialize_vocabulary(w2v_path):
    with codecs.open(w2v_path,"r",encoding="utf-8") as f:
        line=f.readline().strip()
        ss=line.split()
        total=int(ss[0])
        dim=int(ss[1])
        print("total:%d,dim:%d" %(total,dim) )
        vocab=[line.strip().split()[0] for line in f]
        vocab.append('unk')
        #print(vocab)
        #for i,item in enumerate(vocab):
        #    print(i,":",item)
        vocab_dict=dict( [(x,y+1) for (y,x) in enumerate(vocab)])
        #print(vocab_dict)
        return vocab_dict,vocab

def do_load_data(path,sentence_len):
    x=[]
    y=[]
    xlength=[]
    fp=open(path,"r")
    for line in fp.readlines():
        line=line.rstrip()
        if not line:
            continue
        ss=line.split(" ")
        assert(len(ss)==(sentence_len*2))
        lx=[]
        ly=[]
        length=0
        for i in xrange(sentence_len):
            lx.append(int(ss[i]))
            ly.append(int(ss[i+sentence_len]))
            if int(ss[i])>0:
                length+=1
        xlength.append(length)
        x.append(lx)
        y.append(ly)
    fp.close()
    #print(xlength)
    return np.array(x),np.array(y),np.array(xlength)

def batch_load_data(lines,sentence_len):
    x=[]
    y=[]
    xlength=[]
    #print("sentence_len*2:",sentence_len*2)
    for line in lines:
        line=line.strip()
        if not line:
            continue
        ss=line.split(" ")
        assert(len(ss)==(sentence_len*2))
        lx=[]
        ly=[]
        length=0
        for i in xrange(sentence_len):
            lx.append(int(ss[i]))
            ly.append(int(ss[i+sentence_len]))
            if int(ss[i])>0:
                length+=1
        x.append(lx)
        y.append(ly)
        xlength.append(length)
    return np.array(x),np.array(y),np.array(xlength)




def load_graph(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename,"rb") as f:
        graph_def=tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph 
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, input_map=None, return_elements=None, name="prefix", op_dict=None, producer_op_list=None)
    return graph 

def ids_to_words(ids,vob):
    words=[ vob[idx-1] for idx in ids]
    return words

def ids_labels_to_words(ids,vob,lbls):
    global LABELS
    labels=[]
    line=""
    words=[ vob[idx-1] for idx in ids]
    for idx in lbls:
        if idx<15:
            labels.append('O')
        else:
            labels.append(LABELS[idx])
    for (word,label) in zip(words,labels):
        line+=word+"/"+label+" "
    line=line.strip()
    return line
        




def test_evaluate(sess,logits_op,length_op,transMatrix_op,x,tX,tY,xlength,vob,model_out,correct_out):
    logits_,_,transMatrix_=sess.run([logits_op,length_op,transMatrix_op],feed_dict={x:tX })#<45
    # logits_ shape [batch_size,80,16] [3,80,16]
    #print(logits_.shape)
    #print(transMatrix_.shape)
    # transMatrix shape [16,16]
    for unary_scores_,y_,length_ ,input_ in zip(logits_,tY,xlength,tX):
        if length_==0:
            continue
        unary_scores_=unary_scores_[:length_]
        viterbi_sequence,_=tf.contrib.crf.viterbi_decode(unary_scores_,transMatrix_)
        viterbi_sequence=np.asarray(viterbi_sequence)
        #print(viterbi_sequence)
        input_=input_[:length_]
        #print("input:")
        #print(input_)
        #print("viterbi_sequence:")
        #print(viterbi_sequence)
        line=ids_labels_to_words(input_,vob,viterbi_sequence)
        #print(line)
        model_out.write("%s\n" % line)

        y_=y_[:length_]
        correct_line=ids_labels_to_words(input_,vob,y_)
        #print(correct_line)
        correct_out.write("%s\n" % correct_line)


def main(frozen_model_filename,filename,sentence_len,limit,w2v_path,model_out_path,correct_out_path):
    #加载已经将参数固化后的图
    graph=load_graph(frozen_model_filename)

    # We can list operations
    # op.values() gives you a list of tensors it produces
    # op.name gives you the name
    # 输入，输出结点也是opeartion，所以，我们可以得到operation的名字
    #for op in graph.get_operations():
        #print(op.name,op.values)

    # 操作有:prefix/Placeholder/inputs_placeholder
    # 操作有:prefix/Accuracy/predictions
    # 为了预测，我们需要找到我们需要feed的tensor，那么就需要该tensor的名字
    # 注意 prefix/Placeholder/inputs_placeholder 仅仅是操作的名字，prefix/Placeholder/inputs_placeholder:0才是tensor的名字
    x=graph.get_tensor_by_name('prefix/input_placeholder:0')
    logits=graph.get_tensor_by_name('prefix/softmax_1/unary_scores:0')
    length=graph.get_tensor_by_name('prefix/length_1:0')
    transMatrix=graph.get_tensor_by_name('prefix/transitions:0')

    vob_dict,vob=initialize_vocabulary(w2v_path)
    model_out=codecs.open(model_out_path,"w+",encoding="utf-8")
    correct_out=codecs.open(correct_out_path,"w+",encoding="utf-8")
    
    with tf.Session(graph=graph) as sess:
        lines=[]
        count=0
        with codecs.open(filename,"r",encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line:
                    continue
                lines.append(line)
                count+=1
                if count%limit==0:
                    print("processing %d lines" % count)
                    tX,tY,xlength=batch_load_data(lines,sentence_len)
                    test_evaluate(sess,logits,length,transMatrix,x,tX,tY,xlength,vob,model_out,correct_out)
                    lines=[]
        print("processing last %d lines" % len(lines))
        tX,tY,xlength=batch_load_data(lines,sentence_len)
        test_evaluate(sess,logits,length,transMatrix,x,tX,tY,xlength,vob,model_out,correct_out)
        lines=[]
        model_out.close()
        correct_out.close()



if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--word2vec_path",default="word_vectrain_50.txt",type=str, help="word2vec_path")
    parser.add_argument("--frozen_model_filename",default="logs/frozen_model.pb",type=str, help="Frozen model file to import")
    parser.add_argument("--test_data_path",default="conll_word.txt",type=str, help="Frozen model file to import")
    parser.add_argument("--model_out_path",default="model_out.txt",type=str, help="model output path")
    parser.add_argument("--correct_out_path",default="correct_out.txt",type=str, help="correct output path")
    parser.add_argument("--max_sentence_len",default=80,type=str, help="Frozen model file to import")
    parser.add_argument("--batch_limit",default=10000,type=str, help="default 10000")
    args=parser.parse_args()
    sentence_len=int(args.max_sentence_len)
    print("sentence_len:",sentence_len)
    limit=args.batch_limit
    filename=args.test_data_path
    w2v_path=args.word2vec_path
    correct_out_path=args.correct_out_path
    model_out_path=args.model_out_path
    main(args.frozen_model_filename,filename,sentence_len,limit,w2v_path,model_out_path,correct_out_path)
        
    print("finish")












