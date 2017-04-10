#!/usr/bin/python
# -*- coding: UTF-8 -*-

import random
import logging
import os
import sys
import math

import numpy
import gensim
import snap

import time
from functools import partial
from multiprocessing import Value
from multiprocessing.pool import Pool

#print('1')

matches =  Value('i', 0)
num = Value('i', 0)
total = Value('i', 0)

# ---------------------------------网络结点的随机游走序列化：开始---------------------------------------s
def get_random_out(Random_in_node_ID, Tset):  # 获得给定结点ID的一个随机输出结点
    Random_in_node = Tset.GetNI(Random_in_node_ID)
    Random_out_node_th = random.randrange(0, Random_in_node.GetOutDeg())
    Random_out_node_ID = Random_in_node.GetOutNId(Random_out_node_th)
    return Random_out_node_ID

def get_random_walk_road(Random_start_node_ID, MAX_WAIK_LENGTH, Tset):  # 获得给定初始随机结点的一个MAX_WAIK_LENGTH步长的随机游走路径
    Road_list = []
    Random_next_node_ID = Random_start_node_ID

    #while True:
    for x in range(MAX_WAIK_LENGTH):
#        if Random_next_node_ID in Road_list:
#            print(Road_list)
#            return Road_list
        
        Road_list.append(str(Random_next_node_ID))
        Random_next_node_ID = get_random_out(Random_next_node_ID, Tset)  # 获得当前结点的下一个随机游走结点
    return Road_list # 返回随机游走的路径
# -------------------------------------------------------------------------------

# ---------------------------------网络结点结构分布式表达训练：开始---------------------------------------
def train_net2vec_total(Tset, MAX_WAIK_LENGTH, WALK_belta, WINDOW, V_SIZE,MAX_TEST_TIMES):
    class Mynetworks():  # 网络序列化类
        def __init__(self, Tage):
            self.Tage = Tage  # 选择序列化方法的标记，可选参数：1) Random_walk

        def __iter__(self):
            WALK_TIMES = int(Tset.GetEdges()*MAX_TEST_TIMES)
            if self.Tage == 'Random_walk':
                for X in range(WALK_TIMES):
                    yield get_random_walk_road(Tset.GetRndNId(), MAX_WAIK_LENGTH, Tset)  # 构造一个初始结点，然后获取该初始结点的最大随机游走步数
            else:
                print("Wrong!")

    # 训练时的输出信息
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))


    # 训练参数设置
    networks = Mynetworks('Random_walk')  # 迭代返回网络
    net_model = gensim.models.Word2Vec(networks, workers=10, size=V_SIZE, window=WINDOW, min_count=0, cbow_mean=1, sg=0, hs=0)
    return net_model

# ----------------------------------------------------------------------------------------------------
def NetworkModel(filePath, TRY_TIMES, MAX_WAIK_LENGTH, MAX_TEST_TIMES, WALK_belta, WINDOW, V_SIZE):
    All_set = snap.LoadEdgeList(snap.PUNGraph, filePath, 0, 1)  # 载入训练网络
    
    snap.DelSelfEdges(All_set) # 删除自连边
    snap.DelZeroDegNodes(All_set)  # 删除度为0的结点

    for X in range(TRY_TIMES):
        if(os.path.exists(filePath+"_m" + str(MAX_TEST_TIMES) + "_s" + str(V_SIZE)+"_w"+str(WINDOW) + "_t" + str(X) +".vec")):
            continue
                       
        mymodel = train_net2vec_total(All_set, MAX_WAIK_LENGTH, WALK_belta, WINDOW, V_SIZE,MAX_TEST_TIMES)  # 训练结点的分布式表达
        mymodel.wv.save_word2vec_format(filePath+"_m" + str(MAX_TEST_TIMES) + "_s" + str(V_SIZE)+"_w"+str(WINDOW) + "_t" + str(X) +".vec", binary=False)
    
    return

def GetNetworkDegree(filePath):
    All_set = snap.LoadEdgeList(snap.PUNGraph, filePath, 0, 1)  # 载入训练网络
    
    snap.DelSelfEdges(All_set) # 删除自连边
    snap.DelZeroDegNodes(All_set)  # 删除度为0的结点
    degs = dict()
    for NI in All_set.Nodes():
        degs[str(NI.GetId())] = NI.GetDeg()

    f = open(filePath + ".deg", "w")
    for (k,v) in degs.items():
        f.write(str(k)+ "\t" + str(v) + "\r\n")
    f.close()
    
    return degs

def ReadNetworkDegree(filePath):
    if not os.path.exists(filePath + ".deg"):
        return GetNetworkDegree(filePath)

    f = open(filePath + ".deg", "r")
    degs = dict()
    for line in f:
        kv = line.split( )
        if len(kv) != 2:
            continue
        degs[kv[0]] = int(kv[1])

    return degs

def Most_Match(model_a, model_b, x ):
    sim = 0.0
    sim_1 = 0.0
    _y=""
    for y in model_b.vocab:
#        _sim = numpy.dot(gensim.matutils.unitvec(model_a[x]), gensim.matutils.unitvec(model_b[y]))
        _sim = Euler_Similarity(model_a[x], model_b[y])
        if(_sim > sim):
            sim_1 = sim
            sim = _sim
            _y = y

    _x = ""
    sim_b = 0.0
    sim_b1=0.0
#    for y in model_a.vocab:
#        _sim = numpy.dot(gensim.matutils.unitvec(model_b[_y]), gensim.matutils.unitvec(model_a[y]))
#        _sim = Euler_Similarity(model_b[_y], model_a[y])
#        if(_sim > sim_b):
#            sim_b1 = sim_b
#            sim_b = _sim
#            _x = y
    sim_b = sim
    
    num.value += 1
    if num.value %1000 == 0:
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime( time.time() )) + "\trunning matching  " + str(num.value))
 
#    if str(x) != str(_x):
#        return ""

    total.value += 1
    if str(x)==str( _y):
        matches.value += 1

    return x + "\t" + _y + "\t" + str((sim + sim_b) / 2.0)

def Pool_Get_Matches(model_a, model_b):
    pool_size =20
    pool = Pool(pool_size)
    m = partial(Most_Match, model_a, model_b)
    ms = pool.map(m, model_a.vocab)
    pool.close()
    pool.join()
    
    print("matched items count for " + str(matches.value))
    return ms, matches.value

def Euler_Similarity(v1, v2):
    _sim = numpy.linalg.norm(v1 - v2)
    _sim = 1 / (math.log(_sim+1) + 1)
    return _sim

def Match_Model(model_a, model_b):
    matches =  Value('i', 0)
    num = Value('i', 0)
    total = Value('i', 0)
    ms, correct = Pool_Get_Matches(model_a, model_b)
    matches =  Value('i', 0)
    num = Value('i', 0)
    total = Value('i', 0)
    
    return ms, correct

def Write_IU(ms, path):
    f = open(path, 'w')
    for m in ms:
        if m != "":
            f.write(m + "\n")
    f.close()

def Match_Models(FILENAME, TRY_TIMES,MAX_TEST_TIMES, V_SIZE,WINDOW, deg_a, deg_b):
    fpath_a =  FILENAME+"_a.edges"
    fpath_b =  FILENAME+"_b.edges"
    iu = dict()
    for X in range(TRY_TIMES):
        model_a = gensim.models.KeyedVectors.load_word2vec_format(fpath_a+"_m" + str(MAX_TEST_TIMES) + "_s" + str(V_SIZE)+"_w"+str(WINDOW) + "_t"+str(X)+".vec", binary=False) 
        for Y in range(TRY_TIMES):
            model_b = gensim.models.KeyedVectors.load_word2vec_format(fpath_b+"_m" + str(MAX_TEST_TIMES) + "_s" + str(V_SIZE)+"_w"+str(WINDOW) + "_t"+str(Y)+".vec", binary=False)
            ms_XY, correct = Match_Model(model_a, model_b)
            to = len(ms_XY)
            Write_IU(ms_XY, FILENAME+"_m" + str(MAX_TEST_TIMES) + "_s" + str(V_SIZE)+"_w"+str(WINDOW) +  "-" + str(X) + "-" + str(Y) + "-t" + str(to) + "-c" + str(correct) + ".match")
            for ms in ms_XY:
                if ms == "":
                    continue
                
                a, b, sim = ms.split('\t')
                if iu.has_key(a+'\t' + b):
                    iu[a+'\t' + b] = iu[a+'\t' + b] + float(sim)
                else:
                    iu[a+'\t' + b] = float(sim)

    iu = sorted(iu.iteritems(), key=lambda d:d[1], reverse = True)
    iu_a = dict()
    iu_b = dict()
    f_raw = open(FILENAME +"_m" + str(MAX_TEST_TIMES) + "_s" + str(V_SIZE)+"_w"+str(WINDOW) +  "_ti" + str(TRY_TIMES) + "_to" + str(len(iu)) + ".raw.matches", 'w')
    total_m = 0
    total_c = 0
    lines = ""
    for (k, v) in iu:
        f_raw.write(k + "\t" + str(v) + "\n")
        
        u_a, u_b = k.split('\t')
        if(iu_a.has_key(u_a) or iu_b.has_key(u_b)):
            continue

        iu_a[u_a] = u_b
        iu_b[u_b] = u_a
        total_m = total_m + 1
        lines += k + "\t" + str(v) + "\t"
        identical = 0
        if(u_a == u_b):
            identical = 1
        total_c = total_c + identical
        min_deg = min(deg_a[u_a], deg_b[u_b])
        score = v * math.log(min_deg)
        lines += str(identical) + "\t" + str(min_deg) + "\t" + str(score) + "\n"
    f_raw.close()

    f = open(FILENAME +"_m" + str(MAX_TEST_TIMES) + "_s" + str(V_SIZE)+ "_w"+str(WINDOW) + "_ti" + str(TRY_TIMES) + "_to" + str(total_m) + "_tc" + str(total_c) + ".matches", 'w')
    f.write(lines)
    f.close()

    print("Total Identified: " + str(total_m) + ", total correct: " + str(total_c))
    
    return

def Match_Network(FILENAME, TRY_TIMES, MAX_WAIK_LENGTH, MAX_TEST_TIMES, WALK_belta, WINDOW, V_SIZE):
    fpath_a =  FILENAME+"_a.edges"
    fpath_b =  FILENAME+"_b.edges"
    print('train model a from ' + fpath_a)
    NetworkModel(fpath_a,TRY_TIMES, MAX_WAIK_LENGTH, MAX_TEST_TIMES, WALK_belta, WINDOW, V_SIZE)

    print('train model b from ' + fpath_b)
    NetworkModel(fpath_b,TRY_TIMES, MAX_WAIK_LENGTH, MAX_TEST_TIMES, WALK_belta, WINDOW, V_SIZE)
    
    print('cal matches...')
    deg_a = ReadNetworkDegree(fpath_a)
    deg_b = ReadNetworkDegree(fpath_b)
    Match_Models(FILENAME, TRY_TIMES,MAX_TEST_TIMES, V_SIZE,WINDOW, deg_a, deg_b)
    
    return

if __name__ == '__main__':
    
    PATH = "/Users/zhouxp/Documents/!exp_match/real/"
    FILENAMELIST = [PATH+"annonymized"
        #PATH + "sina.annonymized_50000_20_0.33_0.5",
 #       PATH + "5000_20_e0.5",PATH + "5000_20_e0.6",
 #      PATH + "5000_40_e0.6",PATH + "5000_20_e0.6",PATH + "5000_80_e0.6",PATH + "5000_100_e0.6"
 #       PATH + "5000_60_e0.6"
#        PATH + "5000_40_e0.5",PATH + "5000_40_e0.6",       PATH + "5000_40_e0.7",
#        PATH + "5000_60_e0.5",PATH + "5000_60_e0.6",       PATH + "5000_60_e0.7",
#        PATH + "5000_80_e0.5",PATH + "5000_80_e0.6",PATH + "5000_80_e0.7",
#        PATH + "5000_100_e0.5",PATH + "5000_100_e0.6",PATH + "5000_100_e0.7"
        
 #                   PATH + "2000_0.4_e0.4",PATH + "2000_0.4_e0.5",PATH + "2000_0.4_e0.6",PATH + "2000_0.4_e0.7",
#                    PATH + "2000_0.3_e0.4",PATH + "2000_0.3_e0.5",PATH + "2000_0.3_e0.6",PATH + "2000_0.3_e0.7",
#                    PATH + "2000_0.2_e0.4",PATH + "2000_0.2_e0.5",PATH + "2000_0.2_e0.6",PATH + "2000_0.2_e0.7",
#                    PATH + "2000_0.1_e0.4",PATH + "2000_0.1_e0.5",PATH + "2000_0.1_e0.6",PATH + "2000_0.1_e0.7",
#                    PATH + "2000_0.05_e0.4",PATH + "2000_0.05_e0.5",PATH + "2000_0.05_e0.6",PATH + "2000_0.05_e0.7", 

   ]                                
    TRY_TIMES = 1 # parameter t in the paper
    
    for FILENAME in FILENAMELIST:

        # size of S in the paper.
        #when changelist is larger than 10, |S| = [value in changelist] * 50, or [value in changelist]*|F|*50
        changelist = [1]#[4,5,6,7,8,9,10,11]

        # parameter x in the paper
        sizes = [500]#, 1500]#, 300, 500, 700, 900]#, 1000, 1500, 2000]

        for WINDOW in changelist:  # 对固定最大步长，查看游走步数的影响
            for SIZE in sizes:
                # 超参设置
                MAX_WAIK_LENGTH = 50
                MAX_TEST_TIMES = WINDOW
                V_SIZE = SIZE
                belta = 1.0
                Match_Network(FILENAME, TRY_TIMES, MAX_WAIK_LENGTH, MAX_TEST_TIMES , belta, 1, V_SIZE)
    print('end..')
 
