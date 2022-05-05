#!/usr/bin/env python3

import findspark
findspark.init()

import re
import shutil
import sys
import time
from loguru import logger
from operator import add
from pathlib import Path
from pyspark.sql import SparkSession

MAX_ITER = 30
TOLERANCE = 1.0e-6
ALPHA = 0.85
PARALLELISM = 10


def parse_line(line):
    tokens = re.split(r"\s+", line)
    return tokens[0], tokens[1]


def initial_ranks(node, outlinks):
    yield (node, 1.0)
    for n in outlinks:
        yield (n, 1.0)


def compute_contributions(node, out_neighbors, rank):
    yield (node, 0.0)
    out_degree = len(out_neighbors)
    for node in out_neighbors:
        yield (node, rank/out_degree)


def main():
    # check command line arguments
    assert len(sys.argv) == 3, "Args: [input-file] [parallelism]"
    file = Path(sys.argv[1])
    assert file.is_file(), "invalid input file"
    path = "file://" + str(file.absolute())
    PARALLELISM = int(sys.argv[2])
    
    logger.info("setting up spark session")
    begin = time.time()
    spark = SparkSession\
        .builder\
        .appName("pagerank")\
        .master("yarn")\
        .config("spark.executor.instances", PARALLELISM)\
        .config("spark.executor.cores", 1)\
        .config("spark.default.parallelism", PARALLELISM)\
        .getOrCreate()
    setup_time = time.time() - begin
    logger.debug("spark setup time: {}".format(setup_time))
    
    logger.info("reading the input file")
    begin = time.time()
    lines = spark.read.text(path).rdd.map(lambda x: x[0])
    logger.info("formatting the dataset")
    links = lines.map(lambda x: parse_line(x)).distinct().groupByKey().cache()
    read_time = time.time() - begin
    logger.debug("file reading time: {}".format(read_time))
    logger.info("partitions = {}", links.getNumPartitions())
    
    logger.info("assigning initial ranks")
    begin = time.time()
    ranks = links.flatMap(lambda x: initial_ranks(x[0], x[1])).distinct()
    N = ranks.count()  # no. of nodes in the graph
    ranks = ranks.mapValues(lambda x: 1.0/N)
    init_time = time.time() - begin
    logger.debug("initialization time: {}".format(init_time))
    
    logger.info("starting iterations")
    begin = time.time()
    for i in range(MAX_ITER):
        old_ranks = ranks
        contribs = links.join(ranks).flatMap(lambda x: compute_contributions(x[0], x[1][0], x[1][1]))
        ranks = contribs.reduceByKey(add).mapValues(lambda x: ALPHA*x + (1-ALPHA)/N)
        
        S = ranks.values().sum()
        if S < 1:
            leak = (1-S)/N
            ranks = ranks.mapValues(lambda rank: rank + leak)
        
        err = old_ranks.join(ranks).mapValues(lambda x: abs(x[0]-x[1])).values().max()
        logger.info("  iter {}, err = {}".format(i, err))
        if err < TOLERANCE:
            break
    iter_time = time.time() - begin
    total_time = setup_time + read_time + init_time + iter_time
    logger.debug("total iteration time: {}".format(iter_time))
    logger.debug("avg iteration time: {}".format(iter_time/(i+1)))
    logger.debug("total run time: {}".format(total_time))
    
    logger.info("saving result")
    outfile = Path("output")
    if outfile.is_file():
        outfile.unlink()
    if outfile.is_dir():
        shutil.rmtree(outfile.absolute())
    outpath = "file://" + str(outfile.absolute())
    ranks.sortBy(lambda x: x[1], ascending=False).saveAsTextFile(outpath)
    

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, format='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>: <level>{message}</level>')

    main()

