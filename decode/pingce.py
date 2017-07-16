#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
import codecs
import argparse

def tj_head(labels):
    head_ids=[idx for (idx,x) in enumerate(labels) if x=='HED' ]
    return head_ids

def judge_head(std_head,usr_head):
    correct=0
    for id in std_head:
        if id in usr_head:
            correct+=1
    return correct


def evaluate(correct_output_path,model_output_path):
    data_heads=0
    model_heads=0
    correct_heads=0
    with codecs.open(correct_output_path,"r",encoding="utf-8") as fstd:
        with codecs.open(model_output_path,"r",encoding="utf-8") as fusr:
            for stdline,usrline in zip(fstd,fusr):
                line_cor_head=0
                stdline=stdline.strip()
                #print("std:")
                #print(stdline)
                std_labels=[words.split('/')[-1] for words in stdline.split()]
                #print(std_labels)
                std_head=tj_head(std_labels)
                #print("std_head:",std_head)
                data_heads+=len(std_head)
                usrline=usrline.strip()
                #print("usr:")
                #print(usrline)
                usr_labels=[words.split('/')[-1] for words in usrline.split()]
                #print(usr_labels)
                assert(len(std_labels),len(usr_labels))
                usr_head=tj_head(usr_labels)
                #print("usr_head:",usr_head)
                model_heads+=len(usr_head)
                line_cor_head=judge_head(std_head,usr_head)
                correct_heads+=line_cor_head

    #print("data_heads;%d, model_heads:%d, correct_heads:%d" % (data_heads,model_heads,correct_heads))
    return data_heads,model_heads,correct_heads




def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--correct_output_path",default="correct_out.txt", type=str,help="correct output path")
    parser.add_argument("--model_output_path",default="model_out.txt",type=str,help="model output path")
    args=parser.parse_args()
    correct_output_path=args.correct_output_path
    model_output_path=args.model_output_path
    data_heads,model_heads,correct_heads=evaluate(correct_output_path,model_output_path)
    precision=correct_heads/float(model_heads)
    recall=correct_heads/float(data_heads)
    fscore=2*precision*recall/(precision+recall)
    print("data_heads:%d, model_heads:%d, correct_heads:%d, precision:%f, recall:%f, fscore:%f" % (data_heads,model_heads,correct_heads,precision,recall,fscore))



if __name__=='__main__':
    main()



















