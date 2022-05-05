#!/usr/bin/env python3
############################################################
#                  Edited by Ning Li for Para-Opt implementation (V1.0.1)                 #
#Usage:
#1) Implement user-specified algorithm in " user_algo_run(para_set, max_it)";
#2) Specifiy the path of the input data set in "workload";
#3) Specify the SLO of completion time in "time_budget";
#4) Specify the tolerable error (TE)  in "TOLERANCE";
#5) The output of Para-Opt is the optimized parallelism setting in "para_set".
#Assumptions:
#1) Each  partition is allocated the same  # of CPU cores and RAM size;
#2) Oversubmitting wouldn't significantly improve computation performance;
#3) For Spark 3.1.2 in-memory processing.
############################################################
import findspark
findspark.init()

import re
import shutil
import sys
import time
import math
import multiprocessing
from loguru import logger
from operator import add
from pathlib import Path
from pyspark.sql import SparkSession

#MAX_ITER = 30
MAX_ITER = 100
TOLERANCE = 1.0e-8
#TOLERANCE = 1.0e-9
#TOLERANCE = 1.0e-10
ALPHA = 0.85

#The parallsliem limit for the first point;
P0 = 2.0
#The parallsliem limit for the 2nd point;
P1 = 128.0
#The parallsliem limit for the 3rd point;
P2 = 32.0
#The throughput for the first point;
lambda0 = 0.1
#The throughput for the 2nd point;
lambda1 = 0.1
#The throughput for the 3rd point;
lambda2 = 0.1
#The upper limit;
eta = 0.00000001
#small value (used for convergence);
epslan = 10e-12
#nature number value;
EV = 2.718281828459  
#workload;
workload = ""
#for parallelism setting adjustment; 
single_ite_estimation = 0.0 

#the array for setup time under different parallelism;
setup_time_arr = []
#setup_time_num = 0
#the array for the mean iteration time under different parallelism;
iteration_time_arr = []
#iteration_time_num = 0

### Newton calculation;
def tell_valid_gamma(gamma):
    i_ret = 1
    f = (math.pow(EV, -gamma*P0) - math.pow(EV,-gamma*P1))*(P1-1)
    g = (math.pow(EV, -gamma*P0) - math.pow(EV,-gamma*P2))*(P2-1)
    #logger.info("Test P0 = {}  P1 = {}  P2 = {}", P0, P1, P2)
    #logger.info("Test lambda0 = {}  lambda1 = {}  lambda2 = {}", lambda0, lambda1, lambda2)
    #logger.info("Test v1 = {}  v2 = {}  v3 = {}", math.pow(EV, -gamma*P0), math.pow(EV,-gamma*P1), math.pow(EV,-gamma*P2))
    #logger.info("Test f = {}  g = {}  abs(f-g) = {}", f, g, abs(f-g))
    if abs(f-g) < eta:
        i_ret = 0
    return i_ret

#calculate B value;
def B_value(gamma):
    f1 = P1*lambda0/lambda1-P2*lambda0/lambda2
    f2 = (math.pow(EV, -gamma*P0) - math.pow(EV,-gamma*P1))*(P1-1)
    f3 = (math.pow(EV, -gamma*P0) - math.pow(EV,-gamma*P2))*(P2-1)
    f0=f1/P0/(f2-f3);
    return f0;

#derivative funtion of B_value;
def par_B(gamma):
    f1 = (math.pow(EV,-gamma*P0) - math.pow(EV,-gamma*P1))*(P1-1) - (math.pow(EV,-gamma*P0) - math.pow(EV,-gamma*P2))*(P2-1)
    f1 = f1 * P0
    xx = (P2 * lambda0/lambda2 - P1 * lambda0/lambda1) * P0
    yy = (P1 * math.pow(EV,-gamma*P1) - P0 * math.pow(EV,-gamma*P0))*(P1-1)
    zz = (P2 * math.pow(EV,-gamma*P2) - P0 * math.pow(EV,-gamma*P0))*(P2-1)
    f2 = xx * (yy-zz)/f1/f1
    return f2;
               
#The function from defined as:
#f(gamma)=P0+B*P0*(e^(-gamma*P0)-e^(-gamma*P)*(P-1)-P*lmabda(P0)/lmabda(P1)
def fgamma(gamma):
    xx = P0 + B_value(gamma) * P0 * (math.pow(EV,-gamma * P0) - math.pow(EV,-gamma*P1)) * (P1-1) - P1 * lambda0/lambda1
    return xx

#derivative of function fgamma;
def par_fgamma(gamma):
    xx = B_value(gamma) * P0 * (P1 * math.pow(EV,-gamma*P1) - P0 * math.pow(EV,-gamma*P0))*(P1-1)
    yy = P0 * (math.pow(EV,-gamma*P0) - math.pow(EV,-gamma*P1)) * (P1-1) * par_B(gamma)
    zz = xx + yy
    return zz

#throughput function, i.e., lambda(P);
def tput(gamma, P):
    x = P*lambda0/(P0+B_value(gamma) * P0 * (math.pow(EV,-gamma*P0) - math.pow(EV,-gamma*P))*(P-1))
    return x

#Newton calculation;
#type=1: for setup time;
#type=2: for iteration time;
def newton(type):
    #the array for setup time under different parallelism;
    global setup_time_arr
    #global setup_time_num
    #the array for the mean iteration time under different parallelism;
    global iteration_time_arr
    #global iteration_time_num
    #set initial value x
    x = 0.01
    y = 1
    i_ret = 4
    last_thpt = 0.0
    curr_thpt = 0.01
    
    while True:
        #Tell if the gamma is valid;
        if tell_valid_gamma(x) == False:
            return 0
        
        z = fgamma(x)
        if abs(z) < epslan: 
            break
        
        pd = par_fgamma(x)
        if pd > 0.0:
            x = x - fgamma(x)/pd
        
    #logger.info("Test gamma = {}", x)
    #logger.info("Test pd = {}", pd)
    
    for y in range(1, 1025):
        if curr_thpt > last_thpt:
            last_thpt = curr_thpt
            curr_thpt = tput(x,float(y))
            logger.info("Test curr_thpt = {}", curr_thpt)
            #type=1: for setup time;
            if type == 1:
                setup_time_arr.append(curr_thpt)
            #type=2: for iteration time;
            else:
                iteration_time_arr.append(curr_thpt)         
        else:
            break
            
    i_ret = y - 2
    return i_ret

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#>>>>>>>>>>>>Estimation for the number of iterations>>>>>>>>>>>>>>>
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#the index of the iteration which result will be used as 
#the first parameter; 
start_ite_idx = 1
#the three consecutive errors obtained for the iterations indexed by
#start_ite_idx, start_ite_idx+1, start_ite_idx+2
err_arr = [0.00000, 0.00000, 0.00000]
#the number of errors that have been obtained;
#err_num = 0
convergence = 1.0e-7

#The estimation for the number of iterations for expected convergence;
it_n = 0

#the number of nodes;
N = 0;

#The a parameter;
a_para = 0.000
#The b parameter;
b_para = 0.000
#The c parameter;
c_para = 0.000

def validate_err():
    if err_arr[0] * err_arr[1] * err_arr[2] > 0.0:
        return 1
    else:
        return 0
    
def deal_with_c():
    global c_para
    #if validate_err() == 0:
    #    return 0.0
    c_para = (err_arr[2] - err_arr[1])/(err_arr[1] - err_arr[0]);
    return c_para

def deal_with_a():
    global a_para
    #if validate_err() == 0:
    #    return 0.0
    a_para = err_arr[0] + math.pow((err_arr[1] - err_arr[0]), 2.0)/(2 * err_arr[1] - err_arr[0] - err_arr[2]);
    return a_para

def deal_with_b():
    global b_para
    #if validate_err() == 0:
    #    return 0.0
    b_para = (math.pow((err_arr[1] - err_arr[0]), 2.0)/(2 * err_arr[1] - err_arr[0] - err_arr[2]))/math.pow((err_arr[2]-err_arr[1])/(err_arr[1]-err_arr[0]), float(start_ite_idx+1));
    return b_para

def estimation_err(ite):
    return a_para - b_para * math.pow(c_para, float(ite))

def estimation_ite():
    if validate_err() == 0:
        return 0.0
    deal_with_c()
    deal_with_a()
    deal_with_b()
    it = 1
    e_v = estimation_err(it)
    ex_v = estimation_err(1024)
    logger.info("\tthe estimation function for err_arr: a:{} b:{} c:{}".format(a_para, b_para, c_para))
    last_e_v = 0.0;
    #convergence * N/100000
    while abs(e_v - last_e_v) > (TOLERANCE * N / ex_v) * math.pow((TOLERANCE/convergence),2) * (TOLERANCE * N) and it < 1024:
    #while abs(e_v - last_e_v) > math.pow((TOLERANCE/convergence), math.log10(1/(TOLERANCE * N))) * (TOLERANCE * N) and it < 1024:
    #while abs(e_v - last_e_v) > math.pow((TOLERANCE * N), 3) and it < 1024:
        it = it + 1
        last_e_v = e_v
        e_v = estimation_err(it)
        #if it < MAX_ITER:
        #    logger.info("\tErr:{} = {} VS. {}".format(it, e_v, N*TOLERANCE))
    return it
    

########################################


def parse_line(line):
    tokens = re.split(r"\s+", line)
    return tokens[0], tokens[1]


def initial_ranks(node, outlinks):
    yield (node, 1.0)
    for n in outlinks:
        yield (n, 1.0)


def compute_contributions(nodes, rank):
    size = len(nodes)
    for node in nodes:
        yield (node, rank/size)

def user_algo_run(para_set, max_it):
    global N
    global err_arr
    global it_n
    if max_it > MAX_ITER:
        max_it = MAX_ITER
    
    # check command line arguments
    #file = Path(sys.argv[1])
    #file = Path("acm-v8.txt")
    #file = Path("web-Google.txt")
    #file = Path("web-Stanford.txt")
    file = Path(workload)
    assert file.is_file(), "invalid input file"
    path = "file://" + str(file.absolute())
    
    logger.info("setting up spark session")
    #the start time of setup;
    begin = time.time()
    spark = SparkSession\
        .builder\
        .appName("pagerank")\
        .config("spark.executor.instances", para_set)\
        .config("spark.executor.cores", 1)\
        .config("spark.default.parallelism", para_set)\
        .config("spark.executor.memory", "1G")\
        .config("spark.driver.memory", "1G")\
        .getOrCreate()
    
    logger.info("reading the input file")
    lines = spark.read.text(path).rdd.map(lambda x: x[0])
    logger.info("formatting the dataset")
    links = lines.map(lambda x: parse_line(x)).distinct().groupByKey().cache()
    logger.info("spark.default.parallelism = {}", para_set)
    logger.info("partitions = {}", links.getNumPartitions())
    
    #if links.getNumPartitions() !=  para_set:
    #    links.repartition(para_set)
    #    logger.info("partitions = {}", links.getNumPartitions())
    
    logger.info("assigning initial ranks")
    ranks = links.flatMap(lambda x: initial_ranks(x[0], x[1])).distinct()
    N = ranks.count()  # no. of nodes in the graph
    ranks = ranks.mapValues(lambda x: 1.0/N)
    
    partitions = ranks.getNumPartitions()
    
     #the setup time;
    setup_time = time.time() - begin
    logger.debug("spark setup time: {}".format(setup_time))
    
    #the start time of iterations;
    begin = time.time()
    
    #for parallelism setting adjustment; 
    single_ite = 0.0
    
    for i in range(max_it):
        #the start time of iterations;
        ite_begin = time.time()
        ###############################
        old_ranks = ranks
        contribs = links.join(ranks).flatMap(lambda x: compute_contributions(x[1][0], x[1][1]))
        ranks = contribs.reduceByKey(add).mapValues(lambda x: ALPHA*x + (1-ALPHA)/N)
        #ranks = ranks.repartition(partitions)
        
        #the dynamic adjustment for para_set;
        if partitions != para_set:
            ranks = ranks.repartition(para_set)
            #logger.info("\tdynamic adjustment para_set: {} -> {}".format(partitions, para_set))
        else:
            ranks = ranks.repartition(partitions)
        
        S = ranks.values().sum()
        if S < 1:
            leak = (1-S)/N
            ranks = ranks.mapValues(lambda rank: rank + leak)
        
        err = old_ranks.join(ranks).mapValues(lambda x: abs(x[0]-x[1])).values().sum()
        #Recording errors;
        if i >= start_ite_idx and i - start_ite_idx < 3:
            err_arr[i-start_ite_idx] = err
            if it_n == 0:
                logger.info("\trecording err_arr: {} -> {}".format(i, err_arr[i-start_ite_idx]))
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        logger.info("iteration {}: err = {}".format(i, err))
        
        #the end time of iterations;
        ite_end = time.time()
        single_ite = ite_end - ite_begin
        #the dynamic adjustment for para_set;
        if max_it == MAX_ITER:
            #if (single_ite_estimation - single_ite)/single_ite > 0.1:
               # para_set = para_set - 1;
                #if para_set < 2:
                #   para_set = 2
                #logger.info("single_ite_estimation = {}  single_ite = {}  change = {}", single_ite_estimation, single_ite, (single_ite_estimation - single_ite)/single_ite)
            #elif (single_ite - single_ite_estimation)/single_ite > 0.1:
                #para_set = para_set + 1;
            logger.info("\tsingle_ite_estimation = {}  single_ite = {}  change = {}", single_ite_estimation, single_ite, (single_ite_estimation - single_ite)/single_ite)
        else:
            logger.info("\tsingle_ite = {}", single_ite)
       
        ###############################
        if err < N*TOLERANCE:
            break
            
    #Estimate the number of iterations required for the algorithm;
    if it_n == 0:
        it_n = estimation_ite()
        logger.info("\tthe estimation for the number of iterations required for the algorithm: {}".format(it_n))
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    logger.info("saving ranks to 'output'")
    outfile = Path("output")
    if outfile.exists():
        shutil.rmtree(outfile.absolute())
    outpath = "file://" + str(outfile.absolute())
    ranks.sortBy(lambda x: x[1], ascending=False).saveAsTextFile(outpath)
    
    iter_time = time.time() - begin
    total_time = setup_time + iter_time
    logger.info("\ttotal iteration time: {}".format(iter_time))
    logger.info("\tavg iteration time: {}".format(iter_time/(i+1)))
    logger.info("\tavg setup time: {}".format(total_time - iter_time))
    logger.info("\ttotal run time: {}".format(total_time))
    
    arr = [0.1, 0.1, 0.1, 0.1]
    arr[0] = float(setup_time)
    arr[1] = float(iter_time/(i+1))
    arr[2] = float(N)
    arr[3] = single_ite
    
    spark.stop()
    return arr

def main():
    ########################################
    #Edited by Ning Li at 3:47 PM;
    #Test running;
    #Page rank normalized processing speed;
    global P0
    global P1
    global P2
    global lambda0
    global lambda1
    global lambda2
    global TOLERANCE
    global workload
    global single_ite_estimation
    
    P0 = 4.0
    P0_1 = P0 
    P1 = 128.0
    P2 = 32.0
    

    # check input arguments
    assert len(sys.argv) == 4, "Usage: [Data-set-file] [TOLERANCE] [SLO of completion time (S)]"
    workload = sys.argv[1]
    TOLERANCE = float(sys.argv[2])
    time_budget = float(sys.argv[3])
    core_num = multiprocessing.cpu_count()
    
    logger.info(">>> Workload = {}", workload)
    logger.info(">>> TOLERANCE = {}", TOLERANCE)
    logger.info(">>> Time budget = {}", time_budget)
    logger.info(">>> # of CPU Cores = {}", core_num)
    
    setup_time1 = 0.001
    setup_time2 = 0.001
    setup_time3 = 0.001
    
    iteration_time1 = 0.001
    iteration_time2 = 0.001
    iteration_time3 = 0.001
    
    res = user_algo_run(int(P0), 4)
    setup_time1 = res[0]
    if res[1]/res[3] > 1.20:
        res[1] = (4 * res[1] + (it_n - 4) * res[3] * 0.95)/it_n
    iteration_time1 = res[1]
    logger.info("Res<P0> P = {}  setup = {}  iterations = {}  N = {}", P0, res[0], res[1], res[2])
    res = user_algo_run(int(P1), 3)
    setup_time2 = res[0]
    if res[1]/res[3] > 1.20:
        res[1] = (3 * res[1] + (it_n - 3) * res[3] * 0.95)/it_n
    iteration_time2 = res[1]
    logger.info("Res<P1> P = {}  setup = {}  iterations = {}  N = {}", P1, res[0], res[1], res[2])
    res = user_algo_run(int(P2), 3)
    setup_time3 = res[0]
    if res[1]/res[3] > 1.20:
        res[1] = (3 * res[1] + (it_n - 3) * res[3] * 0.95)/it_n
    iteration_time3 = res[1]
    logger.info("Res<P2> P = {}  setup = {}  iterations = {}  N = {}", P2, res[0], res[1], res[2])
    
    #Correct the setup time estimation by skipping the barrier range;
    if setup_time1 > 3 * setup_time2 and setup_time1 > 3 * setup_time3:
        P0 = P0 * 2
        #if 2 * P0 <= P1 and 2 * P0 <= P2: 
        if P0 < P1 and P0 < P2:
            res = user_algo_run(int(P0), 4)
            setup_time1 = res[0]
            #iteration_time1 = res[1]
            logger.info("Again Res<P0> P = {}  setup = {}  iterations = {}  N = {}", P0, res[0], res[1], res[2])
    ################################
    
    lambda0 = 1000.0/setup_time1
    lambda1 = 1000.0/setup_time2
    lambda2 = 1000.0/setup_time3
    
    logger.info("For setup time P0 = {}  P1 = {}  P2 = {}", P0, P1, P2)
    logger.info("For setup time lambda0 = {}  lambda1 = {}  lambda2 = {}", lambda0, lambda1, lambda2)
    
    para_set1 = newton(1)
    logger.info("For setup time paralleliem_set = {}", para_set1)
    
    if P0 > P0_1:
        P0 = P0_1
    
    lambda0 = 1000.0/iteration_time1
    lambda1 = 1000.0/iteration_time2
    lambda2 = 1000.0/iteration_time3
    
    logger.info("For iteration time P0 = {}  P1 = {}  P2 = {}", P0, P1, P2)
    logger.info("For iteration time lambda0 = {}  lambda1 = {}  lambda2 = {}", lambda0, lambda1, lambda2)
    
    para_set2 = newton(2)
    logger.info("For iteration time paralleliem_set = {}", para_set2)
    
    parallelism_set = 2
    min_com_time = 9999999.0
    min_com_idx = 0
    
    while parallelism_set < min(para_set1, para_set2):
        est_com_time = 1000.0/setup_time_arr[parallelism_set-1] + it_n * 1000.0/iteration_time_arr[parallelism_set-1]
        
        logger.info("For Estimated raw time under {} = {} , {}", parallelism_set, 1000.0/setup_time_arr[parallelism_set-1], 1000.0/iteration_time_arr[parallelism_set-1])
        
        logger.info("For Estimated completion time under {} = {} , {} , {}", parallelism_set, est_com_time, 1000.0/setup_time_arr[parallelism_set-1], it_n * 1000.0/iteration_time_arr[parallelism_set-1])
        
        if est_com_time < min_com_time:
            min_com_time = est_com_time
            min_com_idx = parallelism_set
            
        #if est_com_time > time_budget:
        if est_com_time > time_budget and abs(est_com_time - time_budget)/time_budget > 0.05:
            parallelism_set = parallelism_set + 1
        else:
            break
            
    if min_com_idx > core_num:
        min_com_idx = core_num
    
    if parallelism_set >= min(para_set1, para_set2) or parallelism_set > core_num:
        parallelism_set = min_com_idx
        logger.info(">>> Running under the best-effort Spark paralleliem_set = {} ......", parallelism_set)
    else:
        logger.info(">>> Running under the optimized Spark paralleliem_set = {} ......", parallelism_set)
    #for parallelism setting adjustment; 
    single_ite_estimation = 1000.0/iteration_time_arr[parallelism_set-1] 
    res = user_algo_run(parallelism_set, MAX_ITER)
    
    return
    ########################################
    
    
    

if __name__ == "__main__":
    main()
