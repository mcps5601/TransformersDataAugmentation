#!/usr/bin/env bash

SRC=/data/workspace/TransformersDataAugmentation/src
CACHE=/data/workspace/TransformersDataAugmentation/CACHE
TASK=snips
NUMEXAMPLES=10

for i in {0..14};
do
    RAWDATADIR=/data/workspace/TransformersDataAugmentation/datasets/${TASK}/exp_${i}_${NUMEXAMPLES}
    T5DIR=$RAWDATADIR/t5
    mkdir $T5DIR/t5_dev_test
    mv $T5DIR/dev.tsv $T5DIR/t5_dev_test/dev.tsv
    mv $T5DIR/test.tsv $T5DIR/t5_dev_test/test.tsv
done