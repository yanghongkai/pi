#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import codecs

def getFileDir(rootdir):
    l_filename=[]
    for parent,dirnames,filenames in os.walk(rootdir):
        for filename in filenames:
            filename=os.path.join(parent,filename)
            #print filename
            l_filename.append(filename)

    return l_filename


def travDict2(d):
    for fitem in d:
        for sitem in d[fitem]:
            print('dict[%s][%s]=%s' % (fitem,sitem,d[fitem][sitem]))


def travDict3(d):
    for fitem in d:
        for sitem in d[fitem]:
            for titem in d[fitem][sitem]:
                print('dict[%s][%s][%s]=%s' % (fitem,sitem,titem,d[fitem][sitem][titem])) 


def travList(l):
    for item in l:
        print(item,end=' ')

def travList2(l):
    for flist in l:
        for item in flist:
            print(item,end=' ')
        print()







def sortDictKeyAsc(d):
    l_d=sorted(d.items(),key=lambda x: x[0],reverse=False)
    #print l_d 
    # [ ('a',1),('b',3),('c',2)  ]
    #sort_d={}
    #for t_item in l_d:
    #    print t_item
    #    k=t_item[0]
    #    v=t_item[1]
    #    print k,'-->',v
    #    sort_d[k]=v
    #print sort_d
    return l_d


def sortDictVal(d,flag):
    l_d=sorted(d.items(),key=lambda x: x[1],reverse=flag)
    # [('b',3),('c',2),('a',1)]
    return l_d


#a={'a':{'b':5,'c':1,'d':2,'a':4},'c':{'d':1},'b':{'c':2,'a':4}}
def sortDictVal2(d,flag):
    #print d.items() [('a',{'a':4,'c':1,'b':5,'d':2}),('c',{'d':1})]
    for t_item in d.items():
        #print t_item
        kitem=t_item[0]
        d_item=t_item[1]
        #print kitem [('b',5),('a',4),('d',2),('c',1)]
        l_d2=sorted(d_item.items(),key=lambda x: x[1],reverse=flag)
        #print(l_d2)
        for t_freq in l_d2:
            print('dict[%s][%s]=%s' %  (kitem,t_freq[0],t_freq[1]))



def sortDictVal3(d,flag):
    #print(d.items())
    for t_item in d.items():
        #print(t_item) # ('a':{'a':{'yhk':5,'lym':2}})
        kitem1=t_item[0]
        d_item=t_item[1]
        #print(d_item) # {'a':{'yhk':5,'lym':2}}
        for t_item2 in d_item.items():
            #print(t_item2) # ('a':{'yhk':5,'lym':2})
            kitem2=t_item2[0]
            d_item2=t_item2[1]
            l_d2=sorted(d_item2.items(),key=lambda x: x[1],reverse=flag)
            #print l_d2
            for t_freq in l_d2:
                print('dict[%s][%s][%s]=%s' % (kitem1,kitem2,t_freq[0],t_freq[1]))


def merge(filename,out):
    f=codecs.open(filename,"r",encoding="utf-8")
    idx=1
    for line in f:
        print(idx)
        line=line.strip()
        if len(line)>0:
            out.write(line+"\n")
        else:
            continue
        idx+=1
    f.close()



def divideFile(filename,outname,num,coding):
    # num=1000000 200-300M
    f=codecs.open(filename,"r",encoding=coding)
    oriname=outname
    out=codecs.open(outname,"w+",encoding="utf-8")
    idx=1
    count=1
    line=f.readline()
    try:
        while line:
            print(idx)
            line=line.strip()
            idx+=1
            if(idx%num==0):
                out.close()
                outname=oriname+"_"+str(count)
                print(outname)
                out=codecs.open(outname,"w+",encoding="utf-8")
                out.write(line+"\n")
                count+=1
                try:
                    line=f.readline()
                except:
                    print('read error')
            else:
                out.write(line+"\n")
                try:
                    line=f.readline()
                except:
                    print('read error')
    except:
        print('error')

# 在程序当前运行的目录下边产生新的子目录
def mCurDir(dirname):
    current_dir=os.getcwd()
    current_dir=os.path.join(current_dir,dirname)
    #print current_dir
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    return current_dir

# 根据文件全称（带路径）获取文件名字
def getFileName(filepath):
    spl=filepath.split('/')
    idx=len(spl)-1
    filename=spl[idx]
    return filename


# 分隔文件，将一个大文件分隔成几个小文件
def divideWholeDirF(rootdir,dirname,num,coding):
    dir=mCurDir(dirname)
    #print dir
    filenames=getFileDir(rootdir)
    for filename in filenames:
        print(filename)
        cfilename=getFileName(filename)
        #print filename
        outname=os.path.join(dir,cfilename)
        print(outname)
        divideFile(filename,outname,num,coding)



if __name__=='__main__':
    #d={'a':1,'c':2,'b':3}
    #sortDictKeyAsc(d)

    rootdir='/home/yhk/test/divide/test/'
    dirname='small'
    coding='utf-8'
    num=2000000
    divideWholeDirF(rootdir,dirname,num,coding)


    



