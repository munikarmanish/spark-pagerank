#!/usr/bin/env python3

import networkx as nx
import numpy as np
import pandas as pd
import shutil
import sys
import time
from collections import defaultdict
from loguru import logger
from pathlib import Path

ALPHA = 0.85
TOL = 1.0e-9
MAX_ITER = 30


def main():
    # check command line arguments
    file = Path(sys.argv[1])
    assert file.is_file(), "invalid input file"
    
    logger.info("reading input file")
    begin = time.time()
    G = nx.read_edgelist(str(file), create_using=nx.DiGraph)
    read_time = time.time() - begin
    logger.debug("file reading time: {}".format(read_time))
    nodes = list(G)
    N = len(G)  # number of nodes
    
    logger.info("assigning initial ranks")
    begin = time.time()
    for u in G:
        G.nodes[u]['rank'] = 1.0/N
    init_time = time.time() - begin
    logger.debug("ranks initialization time: {}".format(init_time))
    
    logger.info("starting pagerank iterations")
    begin = time.time()
    for i in range(MAX_ITER):
        # save as old
        for u in G:
            G.nodes[u]['old'] = G.nodes[u]['rank']
            G.nodes[u]['rank'] = (1-ALPHA)/N

        # add link contributions
        for u in G:
            n = len(G.out_edges(u))
            if n == 0:
                continue
            x = G.nodes[u]['old'] / n
            for _,v in G.out_edges(u):
                G.nodes[v]['rank'] += ALPHA*x
                # display((v, x))

        # add leaked ranks
        S = sum([G.nodes[u]['rank'] for u in G])
        if S < 1:
            leak = (1-S)/N
            for u in G:
                G.nodes[u]['rank'] += leak

        # display({u:G.nodes[u]['rank'] for u in G})

        x = np.array([G.nodes[u]['old'] for u in G])
        y = np.array([G.nodes[u]['rank'] for u in G])
        err = np.linalg.norm(x-y, ord=1)
        logger.info("  i = {}, err = {}".format(i, err))
        if err < N*TOL:
            break
    iter_time = time.time() - begin
    total_time = read_time + init_time + iter_time
    logger.debug("total iteration time: {}".format(iter_time))
    logger.debug("avg iteration time: {}".format(iter_time/(i+1)))
    logger.debug("total run time: {}".format(total_time))
    
    logger.info("saving result")
    outfile = Path("output")
    if outfile.is_file():
        outfile.unlink()
    if outfile.is_dir():
        shutil.rmtree(outfile.absolute())
    ranksdf = pd.DataFrame([(u, G.nodes[u]['rank']) for u in G], columns=['node', 'pr'])
    ranksdf = ranksdf.sort_values(by='pr', ascending=False)
    ranksdf.to_csv(str(outfile), sep=" ", header=False, index=False)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, format='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>: <level>{message}</level>')
    main()