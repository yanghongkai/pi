# -*- coding: utf-8 -*-

import sys
import os
import codecs
import argparse
import re
import yfile
reload(sys)
sys.setdefaultencoding('utf-8')
print sys.getdefaultencoding()

_SPACE_SPLIT=re.compile(r'\s\s*')

totalLine=0
longLine=0
totalChars=0

class Sentence:
    def __init__(self):
        self.tokens=[]
        self.chars=0
        self.labels={'HED':15, 'SBV':1, 'VOB':2, 'IOB':3, 'FOB':4, 'DBL':5, 'ATT':6, 'ADV':7, 'CMP':8, 'COO':9, 'POB':10, 'LAD':11, 'RAD':12, 'IS':13, 'WP':14}

    def addToken(self,t):
        self.chars+=1
        self.tokens.append(t)

    def clear(self):
        self.tokens=[]
        self.chars=0

    # label -1,unknown
    # 0->'O'
    # 1->'B'
    # 2->'M'
    # 3->'E'
    # 0->填充的
    # 15->HED 1->SBV 2->VOB 3->IOB 4->FOB
    # 5->DBL 6->ATT 7->ADV 8->CMP 9->COO
    # 10->POB 11->LAD 12->RAD 13->IS 14->WP
    def generate_tr_line(self,x,y,vob):
        for t in self.tokens:
            spl=t.split('/')
            token=spl[0]
            head=spl[1]
            #print "token:",token,"head:",head
            #idx=vob.GetWordIndex(token)
            if token in vob:
                idx=vob[token]
            else:
                idx=vob['unk']
            #print "token:",token,"idx:",idx
            if head in self.labels:
                label=self.labels[head]
            else:
                label=0
            #print "token:",token,"idx:",idx,'head:',head,"label:",label
            x.append(idx)
            y.append(label)



def processLine(line,out,vob):
    global totalLine
    global longLine
    global totalChars
    
    line=line.strip()
    words_split=_SPACE_SPLIT.split(line)
    sentence=Sentence()
    for token in words_split:
        sentence.addToken(token)
    #print(sentence.chars)
    nn=sentence.chars# sentence length
    x=[]
    y=[]
    totalChars+=sentence.chars
    sentence.generate_tr_line(x,y,vob)
    assert(nn==len(y))
    newline=""
    for i in xrange(nn):
        if i>0:
            newline+=" "
        newline+=str(x[i])+"/"+str(y[i])
    #print(newline)
    out.write("%s\n" % (newline))

    totalLine+=1
    sentence.clear()

    
 
def initialize_vocabulary(w2v_path):
    with codecs.open(w2v_path,"r",encoding="utf-8") as f:
        line=f.readline().strip()
        ss=line.split()
        total=int(ss[0])
        dim=int(ss[1])
        print("total:%d,dim:%d" %(total,dim) )
        #vocab=[]
        #idx=1
        #line=f.readline()
        #while line:
        #    #print(idx)
        #    line=line.strip()
        #    #print("idx:%d,words:%s" % (idx,line.split()[0]))
        #    vocab.append(line.split()[0])
        #    try: 
        #        line=f.readline()
        #        idx+=1
        #    except:
        #        print('can not read')
        #        print("idx:%d,words:%s" % (idx,line.split()[0]))

        vocab=[line.strip().split()[0] for line in f]
        vocab.append('unk')
        #print(vocab)
        vocab_dict=dict( [(x,y+1) for (y,x) in enumerate(vocab)])
        #print(vocab_dict)
        #for key in vocab_dict:
        #    print(key,'->',vocab_dict[key])
        return vocab_dict



def main(argc,argv):
    parser=argparse.ArgumentParser()
    parser.add_argument("--word2vec_path", default="word2vec.txt", type=str, help="word2vec_path")
    parser.add_argument("--data_dir",default="data",type=str,help="data dir")
    parser.add_argument("--output_path",default="outids.txt",type=str,help="output")
    args=parser.parse_args()
    w2v_path=args.word2vec_path
    data_dir=args.data_dir
    print(w2v_path)
    vob=initialize_vocabulary(w2v_path)
    
    out=codecs.open(args.output_path,"w+",encoding="utf-8:")
    filenames=yfile.getFileDir(data_dir)
    for filename in filenames:
        print filename
        f=codecs.open(filename,"r",encoding="utf-8")
        #for line in f.readlines():
        for line in f:
            line=line.strip()
            #print line
            processLine(line,out,vob)
        f.close()
    out.close()
    print("total:%d, long lines:%d, chars:%d " % (totalLine,longLine,totalChars))

if __name__=='__main__':
    main(len(sys.argv),sys.argv)





