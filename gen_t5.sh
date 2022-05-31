for gen_model_name in /data/workspace/Fewshot/experiments/t5-small-mnli-e10_bs32_lr2e-5/checkpoint-34500 t5-large
do
    for task in stsa snips trec
    do
        python src/bert_aug/t5.py --task $task --gen_model_name $gen_model_name
    done
done