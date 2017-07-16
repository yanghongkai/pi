#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
import tensorflow as tf
import argparse
import numpy as np
import codecs
import yfile
import math

buckets=[10]
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

def pad_data(line,pad_sentence_len):
    #x=[]
    for i in xrange(pad_sentence_len-len(line)):
        line.append(0)
    #x.append(line)
    #return np.array(x)
    return line

def process_window(words,sentence_len,ahead_len,behind_len):
    windows=[]
    for idx,word in enumerate(words):
        if idx>ahead_len and idx+behind_len<sentence_len:
            idx_window=words[idx-ahead_len:idx+behind_len+1]
            windows.append(idx_window)
        elif idx<=ahead_len:
            idx_window=words[:ahead_len+behind_len+1] #[0:10]->0-9
            windows.append(idx_window)
        else:
            idx_window=words[sentence_len-ahead_len-behind_len-1:]
            windows.append(idx_window)
    return np.array(windows,dtype=np.int32)





def load_graph(frozen_graph_filename,graph_name):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename,"rb") as f:
        graph_def=tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph 
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, input_map=None, return_elements=None, name=graph_name, op_dict=None, producer_op_list=None)
        a=tf.constant(2.0,name="test_constant")
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

def ids_p_to_words(ids,vob,lbls):
    line=""
    maxidx=lbls.index(max(lbls))
    words=[ vob[idx-1] for idx in ids]
    for i,lbl in enumerate(lbls):
        if i==maxidx:
            #print("max i:",i)
            line+=words[i]+"/"+LABELS[-1]+" "
        else:
            line+=words[i]+"/"+"O"+" "
    line=line.strip()
    return line


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


def test_evaluate(sess,logits_op,x,tX,words_ids,labels,xlength,vob,model_out,correct_out,n_step,n_classes,ahead_len,behind_len,pad_sentence_len):
    logits_=sess.run(logits_op,feed_dict={x:tX })
    # logits_ shape [20*10,16]
    logits_=np.reshape(logits_,(-1,n_step,n_classes))
    #print(logits_.shape)
    # logits_ shape [20,10,16]
    plabels=[]
    total_p=[ [] for j in range(xlength[0])]
    #[ [], [], [], [], [], [], [], [], [] ...]
    for i in range(xlength[0]):
        if i<=ahead_len:
            idx2=i%n_step
            end=min(xlength[0],ahead_len+behind_len+1)
            for j in range(end):
                total_p[j].append(logits_[i][j][-1])
        elif i>ahead_len and i+behind_len<pad_sentence_len:
            idx2=ahead_len
            end=min(xlength[0],i+behind_len+1)
            for j in range(i-ahead_len,end):
                start=i-ahead_len
                total_p[j].append(logits_[i][j-start][-1])
        else:
            idx2=i%n_step
            end=min(xlength[0],pad_sentence_len)
            for j in range(pad_sentence_len-ahead_len-behind_len-1,end):
                start=pad_sentence_len-ahead_len-behind_len-1
                total_p[j].append(logits_[i][j-start][-1])
        p=logits_[i][idx2][-1]
        plabels.append(p)
    #print(total_p)
    
    #for i,ps in enumerate(total_p):
    #    print("i:",i,"len:",len(ps))
    plabels_aver=[]
    for ps in total_p:
        sum=0.0
        for j in ps:
            sum+=j
        average=sum/len(ps)
        #print("aver:",average)
        plabels_aver.append(average)
    #words=ids_to_words(words_ids,vob)
    #for w,p in zip(words,plabels):
    #    print("%s:%f" % (w,p))
    #for w,p in zip(words,plabels_aver):
    #    print("%s:%f" % (w,p))
        
    #print(plabels)
    #print(len(plabels))
    #print(len(words_ids))
    model_line=ids_p_to_words(words_ids,vob,plabels_aver)
    #print(model_line)
    model_out.write("%s\n" % model_line)
    
    correct_line=ids_labels_to_words(words_ids,vob,labels)
    #print(correct_line)
    correct_out.write("%s\n" % correct_line)

def model(frozen_model_filename,graph_name):
    #加载已经将参数固化后的图
    graph=load_graph(frozen_model_filename,graph_name)
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
    x=graph.get_tensor_by_name(graph_name+'/input_placeholder:0')
    logits=graph.get_tensor_by_name(graph_name+'/infer_head:0')
    # logits=graph.get_tensor_by_name(graph_name+'/softmax_1/unary_scores:0')
    # length=graph.get_tensor_by_name(graph_name+'/length_1:0')
    # transMatrix=graph.get_tensor_by_name(graph_name+'/transitions:0')

    sess=tf.Session(graph=graph)
    return sess,x,logits

def test(sess):
    a_=sess.run(a)
    print(a)

def main(sess_buckets,filename,limit,w2v_path,model_out_path,correct_out_path,n_step,n_classes):

    vob_dict,vob=initialize_vocabulary(w2v_path)
    model_out=codecs.open(model_out_path,"w+",encoding="utf-8")
    correct_out=codecs.open(correct_out_path,"w+",encoding="utf-8")
    print(sess_buckets)
        
    
    models=[]
    for bucket in buckets:
       frozen_model_filename=sess_buckets[str(bucket)]
       graph_name="prefix_"+str(bucket)
       sess,x,logits=model(frozen_model_filename,graph_name)
       models.append([sess,x,logits])

    model_id=0
    for idx,bucket in enumerate(buckets):
        if bucket==n_step:
            model_id=idx
    total_lines=0
    long_lines=0
    behind_len=(n_step-1)/2
    ahead_len=behind_len+1
    with codecs.open(filename,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            xlength=[]
            words=[int(w.split("/")[0]) for w in line.split()]
            labels=[int(w.split("/")[-1]) for w in line.split()]
            line_len=len(words)
            if line_len>30:
                long_lines+=1
                continue
            xlength.append(line_len)
            #print('xlength:',xlength)
            #for model_id,model_size in enumerate(buckets):
             #   if line_len<=model_size:
                    #print("model_id:",model_id)
                    #print("model_size:",model_size)
            epoch=math.ceil(line_len/float(n_step))
            epoch=int(epoch)
            pad_sentence_len=epoch*n_step
            pad_x=pad_data(words,pad_sentence_len)
            #pad_y=pad_data(labels,pad_sentence_len)
            #print("pad_x:",pad_x)
            tX=process_window(pad_x,pad_sentence_len,ahead_len,behind_len)
            #print("tX:",tX)
            #print(tX.dtype)
            #print("tX.shape:",tX.shape)
            #print("words len:",len(words))

            sess_op,x_op,logits_op=models[model_id]
            test_evaluate(sess_op,logits_op,x_op,tX,words,labels,xlength,vob,model_out,correct_out,n_step,n_classes,ahead_len,behind_len,pad_sentence_len)
            total_lines+=1

    #print("total_lines:%d, long_lines:%d" % (total_lines,long_lines))
    print("deal_lines:%d, long_lines:%d" % (total_lines,long_lines))

            
            



if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--word2vec_path",default="word_vectrain_50.txt",type=str, help="word2vec_path")
    parser.add_argument("--frozen_model_folder",default="logs/frozen_model.pb",type=str, help="Frozen model file to import")
    parser.add_argument("--test_data_path",default="conll_word.txt",type=str, help="Frozen model file to import")
    parser.add_argument("--model_out_path",default="model_out.txt",type=str, help="model output path")
    parser.add_argument("--correct_out_path",default="correct_out.txt",type=str, help="correct output path")
    parser.add_argument("--n_step",default=10,type=int, help="n_step")
    parser.add_argument("--n_classes",default=16,type=int, help="n_classes")
    parser.add_argument("--batch_limit",default=10000,type=int, help="default 10000")
    args=parser.parse_args()
    limit=args.batch_limit
    n_step=args.n_step
    filename=args.test_data_path
    w2v_path=args.word2vec_path
    n_classes=args.n_classes
    correct_out_path=args.correct_out_path
    model_out_path=args.model_out_path
    model_folder=args.frozen_model_folder
    model_filenames=yfile.getFileDir(model_folder)
    print(model_filenames)
    sess_buckets={}
    for bucket in buckets:
        for model_filename in model_filenames:
            if model_filename.find(str(bucket))>0:
                sess_buckets[str(bucket)]=model_filename
    main(sess_buckets,filename,limit,w2v_path,model_out_path,correct_out_path,n_step,n_classes)
        
    print("finish")












