# Optimizing PageRank on Spark

This project implements distributed PageRank algorithm on Spark using PySpark
and provides a framework for optimizing Spark resource allocation based on
user's SLO requirements.

## Authors

- Manish Munikar
- Ning Li

## Dependencies

- spark (with PySpark support)
- python >= 3.6
- networkx
- pandas
- numpy
- loguru

## Input graph format

Input graph should be a text file in edge-list format: each line should have two
identifiers (from-node and to-node) separated by whitespace. For eg:

    B C
    C B
    D A
    D B
    E B
    E D
    E F
    F B
    F E
    G F
    H F
    I B
    I E
    J B
    J E
    K B
    K E

## How to run?

### How to run serial PageRank?

    python3 pagerank_serial.py <input-file>

For eg:

    python3 pagerank_serial.py web-Stanford.txt

### How to run distributed PageRank?

    python3 pagerank_spark.py <input-file> <no-of-executors>

For eg:

    python3 pagerank_spark.py web-Stanford.txt 10

### How to run Para-Opt?

    python3 para_opt.py <input-file> <tolerance> <SLO-completion-time-in-sec>

For eg:

    python3 para_opt.py web-Google.txt 1e-9 180
