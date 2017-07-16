#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
import tensorflow as tf
import argparse
import numpy as np
import codecs
import yfile

buckets=[10,20,30,40,50,60,70,80]
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

def pad_data(line,sentence_len):
    x=[]
    for i in xrange(sentence_len-len(line)):
        line.append(0)
    x.append(line)
    return np.array(x)


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
        
def test_evaluate_batch(sess,logits_op,length_op,transMatrix_op,x,tX,tY,xlength,vob,model_out,correct_out):
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


def test_evaluate(sess,logits_op,length_op,transMatrix_op,x,tX,tY,xlength,vob,model_out,correct_out):
    logits_,_,transMatrix_=sess.run([logits_op,length_op,transMatrix_op],feed_dict={x:tX })#<45
    # logits_ shape [batch_size,80,16] [3,80,16]
    #print(logits_.shape)
    #print(transMatrix_.shape)
    # transMatrix shape [16,16]
    #print('logits_.shape:',logits_.shape)
    #print('tY.shape:',tY.shape)
    #print('tX.shape:',tX.shape)
    #print('xlength:',xlength)
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
        print(correct_line)
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
    logits=graph.get_tensor_by_name(graph_name+'/softmax_1/unary_scores:0')
    length=graph.get_tensor_by_name(graph_name+'/length_1:0')
    transMatrix=graph.get_tensor_by_name(graph_name+'/transitions:0')

    sess=tf.Session(graph=graph)
    return sess,x,logits,length,transMatrix

def test(sess):
    a_=sess.run(a)
    print(a)

def main(sess_buckets,filename,sentence_len,limit,w2v_path,model_out_path,correct_out_path):

    vob_dict,vob=initialize_vocabulary(w2v_path)
    model_out=codecs.open(model_out_path,"w+",encoding="utf-8")
    correct_out=codecs.open(correct_out_path,"w+",encoding="utf-8")
    #sess,x,logits,length,transMatrix=model(frozen_model_filename)
    print(sess_buckets)
        

    # frozen_model_filename=sess_buckets[str(10)]
    # graph_name_10="prefix10"
    # graph_10=load_graph(frozen_model_filename,graph_name_10)
    # x_10=graph_10.get_tensor_by_name(graph_name_10+'/input_placeholder:0')
    # logits_10=graph_10.get_tensor_by_name(graph_name_10+'/softmax_1/unary_scores:0')
    # length_10=graph_10.get_tensor_by_name(graph_name_10+'/length_1:0')
    # transMatrix_10=graph_10.get_tensor_by_name(graph_name_10+'/transitions:0')
    # sess_10=tf.Session(graph=graph_10)

    # frozen_model_filename=sess_buckets[str(20)]
    # graph_name_20="prefix20"
    # graph_20=load_graph(frozen_model_filename,graph_name_20)
    # x_20=graph_20.get_tensor_by_name(graph_name_20+'/input_placeholder:0')
    # logits_20=graph_20.get_tensor_by_name(graph_name_20+'/softmax_1/unary_scores:0')
    # length_20=graph_20.get_tensor_by_name(graph_name_20+'/length_1:0')
    # transMatrix_20=graph_20.get_tensor_by_name(graph_name_20+'/transitions:0')
    # sess_20=tf.Session(graph=graph_20)

    # frozen_model_filename=sess_buckets[str(30)]
    # graph_name_30="prefix30"
    # graph_30=load_graph(frozen_model_filename,graph_name_30)
    # x_30=graph_30.get_tensor_by_name(graph_name_30+'/input_placeholder:0')
    # logits_30=graph_30.get_tensor_by_name(graph_name_30+'/softmax_1/unary_scores:0')
    # length_30=graph_30.get_tensor_by_name(graph_name_30+'/length_1:0')
    # transMatrix_30=graph_30.get_tensor_by_name(graph_name_30+'/transitions:0')
    # sess_30=tf.Session(graph=graph_30)

    # frozen_model_filename=sess_buckets[str(40)]
    # graph_name_40="prefix40"
    # graph_40=load_graph(frozen_model_filename,graph_name_40)
    # x_40=graph_40.get_tensor_by_name(graph_name_40+'/input_placeholder:0')
    # logits_40=graph_40.get_tensor_by_name(graph_name_40+'/softmax_1/unary_scores:0')
    # length_40=graph_40.get_tensor_by_name(graph_name_40+'/length_1:0')
    # transMatrix_40=graph_40.get_tensor_by_name(graph_name_40+'/transitions:0')
    # sess_40=tf.Session(graph=graph_40)

    # frozen_model_filename=sess_buckets[str(50)]
    # graph_name_50="prefix50"
    # graph_50=load_graph(frozen_model_filename,graph_name_50)
    # x_50=graph_50.get_tensor_by_name(graph_name_50+'/input_placeholder:0')
    # logits_50=graph_50.get_tensor_by_name(graph_name_50+'/softmax_1/unary_scores:0')
    # length_50=graph_50.get_tensor_by_name(graph_name_50+'/length_1:0')
    # transMatrix_50=graph_50.get_tensor_by_name(graph_name_50+'/transitions:0')
    # sess_50=tf.Session(graph=graph_50)

    # frozen_model_filename=sess_buckets[str(60)]
    # graph_name_60="prefix60"
    # graph_60=load_graph(frozen_model_filename,graph_name_60)
    # x_60=graph_60.get_tensor_by_name(graph_name_60+'/input_placeholder:0')
    # logits_60=graph_60.get_tensor_by_name(graph_name_60+'/softmax_1/unary_scores:0')
    # length_60=graph_60.get_tensor_by_name(graph_name_60+'/length_1:0')
    # transMatrix_60=graph_60.get_tensor_by_name(graph_name_60+'/transitions:0')
    # sess_60=tf.Session(graph=graph_60)

    # frozen_model_filename=sess_buckets[str(70)]
    # graph_name_70="prefix70"
    # graph_70=load_graph(frozen_model_filename,graph_name_70)
    # x_70=graph_70.get_tensor_by_name(graph_name_70+'/input_placeholder:0')
    # logits_70=graph_70.get_tensor_by_name(graph_name_70+'/softmax_1/unary_scores:0')
    # length_70=graph_70.get_tensor_by_name(graph_name_70+'/length_1:0')
    # transMatrix_70=graph_70.get_tensor_by_name(graph_name_70+'/transitions:0')
    # sess_70=tf.Session(graph=graph_70)

    # frozen_model_filename=sess_buckets[str(80)]
    # graph_name_80="prefix80"
    # graph_80=load_graph(frozen_model_filename,graph_name_80)
    # x_80=graph_80.get_tensor_by_name(graph_name_80+'/input_placeholder:0')
    # logits_80=graph_80.get_tensor_by_name(graph_name_80+'/softmax_1/unary_scores:0')
    # length_80=graph_80.get_tensor_by_name(graph_name_80+'/length_1:0')
    # transMatrix_80=graph_80.get_tensor_by_name(graph_name_80+'/transitions:0')
    # sess_80=tf.Session(graph=graph_80)










    sess_10,x_10,logits_10,length_10,transMatrix_10=model(sess_buckets[str(10)],"prefix10")
    sess_20,x_20,logits_20,length_20,transMatrix_20=model(sess_buckets[str(20)],"prefix20")
    sess_30,x_30,logits_30,length_30,transMatrix_30=model(sess_buckets[str(30)],"prefix30")
    sess_40,x_40,logits_40,length_40,transMatrix_40=model(sess_buckets[str(40)],"prefix40")
    sess_50,x_50,logits_50,length_50,transMatrix_50=model(sess_buckets[str(50)],"prefix50")
    sess_60,x_60,logits_60,length_60,transMatrix_60=model(sess_buckets[str(60)],"prefix60")
    sess_70,x_70,logits_70,length_70,transMatrix_70=model(sess_buckets[str(70)],"prefix70")
    sess_80,x_80,logits_80,length_80,transMatrix_80=model(sess_buckets[str(80)],"prefix80")
    #print("default graph:",tf.get_default_graph())
    
    
    #models=[]

    #for bucket in buckets:
    #    frozen_model_filename=sess_buckets[str(bucket)]
    #    print(frozen_model_filename)
    #    graph_name="graph_"+str(bucket)
    #    print(graph_name)
    #    sess,x,logits,length,transMatrix=model(frozen_model_filename,graph_name)
    #    models.append([sess,x,logits,length,transMatrix])

    #print(models)
    #print(len(models))
    
    with codecs.open(filename,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            xlength=[]
            words=[int(w.split("/")[0]) for w in line.split()]
            labels=[int(w.split("/")[-1]) for w in line.split()]
            line_len=len(words)
            xlength.append(line_len)
            print('xlength:',xlength)
            for model_id,model_size in enumerate(buckets):
                if line_len<=model_size:
                    print("model_id:",model_id)
                    print("model_size:",model_size)
                    tX=pad_data(words,model_size)
                    tY=pad_data(labels,model_size)
                    #print(tX.shape)
                    #test_evaluate(sess,logits,length,transMatrix,x,tX,tY,xlength,vob,model_out,correct_out)
                    #test_evaluate(sess,logits,length,transMatrix,x,tX,tY,xlength,vob,model_out,correct_out)
                    #sess_op,x_op,length_op,logits_op,transMatrix_op=models[model_id]
                    #test_evaluate(sess_op,logits_op,length_op,transMatrix_op,x_op,tX,tY,xlength,vob,model_out,correct_out)
                    if model_size==10:
                        test_evaluate(sess_10,logits_10,length_10,transMatrix_10,x_10,tX,tY,xlength,vob,model_out,correct_out)
                        #pass
                    elif model_size==20:
                        test_evaluate(sess_20,logits_20,length_20,transMatrix_20,x_20,tX,tY,xlength,vob,model_out,correct_out)
                        #pass
                    elif model_size==30:
                        test_evaluate(sess_30,logits_30,length_30,transMatrix_30,x_30,tX,tY,xlength,vob,model_out,correct_out)
                        #pass
                    elif model_size==40:
                        test_evaluate(sess_40,logits_40,length_40,transMatrix_40,x_40,tX,tY,xlength,vob,model_out,correct_out)
                        #pass
                    elif model_size==50:
                        test_evaluate(sess_50,logits_50,length_50,transMatrix_50,x_50,tX,tY,xlength,vob,model_out,correct_out)
                        #pass
                    elif model_size==60:
                        test_evaluate(sess_60,logits_60,length_60,transMatrix_60,x_60,tX,tY,xlength,vob,model_out,correct_out)
                        #pass
                    elif model_size==70:
                        test_evaluate(sess_70,logits_70,length_70,transMatrix_70,x_70,tX,tY,xlength,vob,model_out,correct_out)
                        #pass
                    elif model_size==80:
                        test_evaluate(sess_80,logits_80,length_80,transMatrix_80,x_80,tX,tY,xlength,vob,model_out,correct_out)
                        #pass
                    else:
                        pass


                    break

            
            
            #lines.append(line)
            #count+=1
            #if count%limit==0:
            #    print("processing %d lines" % count)
            #    tX,tY,xlength=batch_load_data(lines,sentence_len)
            #    test_evaluate(sess,logits,length,transMatrix,x,tX,tY,xlength,vob,model_out,correct_out)
            #    lines=[]
    #print("processing last %d lines" % len(lines))
    #tX,tY,xlength=batch_load_data(lines,sentence_len)
    #test_evaluate(sess,logits,length,transMatrix,x,tX,tY,xlength,vob,model_out,correct_out)
    #lines=[]
    #model_out.close()
    #correct_out.close()



if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--word2vec_path",default="word_vectrain_50.txt",type=str, help="word2vec_path")
    parser.add_argument("--frozen_model_folder",default="logs/frozen_model.pb",type=str, help="Frozen model file to import")
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
    model_folder=args.frozen_model_folder
    model_filenames=yfile.getFileDir(model_folder)
    print(model_filenames)
    sess_buckets={}
    for bucket in buckets:
        for model_filename in model_filenames:
            if model_filename.find(str(bucket))>0:
                sess_buckets[str(bucket)]=model_filename
    main(sess_buckets,filename,sentence_len,limit,w2v_path,model_out_path,correct_out_path)
        
    print("finish")












