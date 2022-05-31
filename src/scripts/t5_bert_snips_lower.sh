#!/usr/bin/env bash

SRC=/data/workspace/TransformersDataAugmentation/src
CACHE=/data/workspace/TransformersDataAugmentation/CACHE
TASK=snips
exp_name=t5_generated/t5-small-mnli-e10_bs32_lr2e-5_snips_10N_top-k_40

for NUMEXAMPLES in 10;
do
    for i in {0..14};
    do
        RAWDATADIR=/data/workspace/TransformersDataAugmentation/$exp_name/exp_${i}_${NUMEXAMPLES}
        T5DIR=$RAWDATADIR/t5
        # cp $RAWDATADIR/test.tsv $T5DIR/test.tsv
        # cp $RAWDATADIR/dev.tsv $T5DIR/dev.tsv
        python $SRC/bert_aug/bert_classifier.py --task $TASK --data_dir $T5DIR --seed ${i} --cache $CACHE  > $RAWDATADIR/bert_t5.log
    done
done