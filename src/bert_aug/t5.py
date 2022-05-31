import argparse
import json
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from tqdm import tqdm, trange
from t5_utils import load_generation_model, load_filtering_model


def prepare_data(sentences: list, prefix: str):
    t5_input_sents = [f"{prefix}: {s}" for s in sentences]

    return t5_input_sents


def gen_t5_data(
    sentences,
    t5_tokenizer,
    t5_model,
    batch_size,
    decoding_strategy,
    do_sample,
    num_generate_per_sentence,
):
    generated_sentences = []
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch_sentences = sentences[i:i+batch_size]
        batch = t5_tokenizer(
            batch_sentences,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).input_ids
        batch = batch.to(device)
        with torch.no_grad():
            if decoding_strategy == 'top-k':
                model_output = t5_model.generate(
                    batch,
                    max_length=128,
                    top_k=decoding_value,
                    do_sample=do_sample,
                    num_return_sequences=num_generate_per_sentence,
                )
            elif decoding_strategy == 'top-p':
                t5_model.config.top_k = None
                model_output = t5_model.generate(
                    batch,
                    max_length=128,
                    top_p=decoding_value,
                    do_sample=do_sample,
                    num_return_sequences=num_generate_per_sentence,
                )
            elif decoding_strategy == 'beam':
                model_output = t5_model.generate(
                    batch,
                    max_length=128,
                    num_beams=decoding_value,
                    do_sample=do_sample,
                    num_return_sequences=num_generate_per_sentence,
                )
            result = t5_tokenizer.batch_decode(
                model_output, skip_special_tokens=True)
            generated_sentences.extend(result)
    print("Finished generating data.")
    return generated_sentences


def filter_sentences(
    origin_sentence,
    new_sentence,
    tokenizer,
    model,
    device,
    filter_mode,
):
    data_encoding = tokenizer(
        origin_sentence,
        new_sentence,
        padding=True,
        # truncation='only_first',
        truncation=True,
        return_tensors='pt'
    ).to(device)
    with torch.no_grad():
        outputs = model(**data_encoding).logits
        predict_proba = nn.functional.softmax(outputs, dim=-1)
        # best_choice = predict_proba[:, 0].argmax()

        # 觀察是否有任何一個 sentence 會被預測為 2 (entailment)
        # roberta-large-mnli: {"0": "CONTRADICTION", "1": "NEUTRAL", "2": "ENTAILMENT"}
        predictions = predict_proba.argmax(dim=-1)
        if filter_mode == 'E_only':
            got_passed = torch.any(predictions == 2).cpu().numpy()
            best_choice = predict_proba[:, 2].argmax().item()
        elif filter_mode == 'EorN':
            got_passed = torch.any(predictions != 0).cpu().numpy()
            if got_passed:
                if torch.any(predictions == 2).cpu().numpy():
                    best_choice = predict_proba[:, 2].argmax().item()
                else:
                    best_choice = predict_proba[:, 1].argmax().item()
            elif not got_passed:
                best_choice = -1
                print(f"Ori: {origin_sentence[0]}")
                print(f"Wrong: {new_sentence[predict_proba[:, 2].argmax()]}")
        elif filter_mode == 'C_only':
            got_passed = torch.any(predictions == 0).cpu().numpy()
            best_choice = predict_proba[:, 0].argmax().item()
        elif filter_mode == 'N_only':
            got_passed = torch.any(predictions == 1).cpu().numpy()
            best_choice = predict_proba[:, 1].argmax().item()

    return best_choice, got_passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu_id",
        default=0,
        type=int
    )
    parser.add_argument(
        "--task",
        default='snips',
        choices=['stsa', 'snips', 'trec'],
        type=str,
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int
    )
    parser.add_argument(
        "--gen_model_name",
        default="t5-large",
        # default="/data/workspace/Fewshot/experiments/t5-small-mnli-e10_bs32_lr2e-5/checkpoint-34500",
    )
    parser.add_argument(
        "--cls_model_name",
        default="roberta-large-mnli",
    )
    parser.add_argument(
        "--decoding_strategy",
        default='top-k',
        choices=['greedy', 'top-k', 'top-p', 'beam']
    )
    parser.add_argument(
        "--topp_value",
        default=0.9,
        type=float
    )
    parser.add_argument(
        "--topk_value",
        default=40,
        type=int
    )
    parser.add_argument(
        "--beam_size",
        default=5,
        type=int
    )
    parser.add_argument(
        "--num_generate_per_sentence",
        default=10,
        type=int,
        help="一開始總共產生幾個句子",
    )
    parser.add_argument(
        "--filter_mode",
        default='EorN',
        choices=['E_only', 'EorN', 'N_only']
    )
    parser.add_argument(
        "--num_exps",
        default=15,
        type=int
    )
    args = parser.parse_args()
    do_sample = True if args.decoding_strategy in ['top-k', 'top-p'] else False

    device = torch.device(
        f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    )
    if args.decoding_strategy == 'top-p':
        decoding_value = args.topp_value
    elif args.decoding_strategy == 'top-k':
        decoding_value = args.topk_value
    elif args.decoding_strategy == 'beam':
        decoding_value = args.beam_size

    t5_tokenizer, t5_model = load_generation_model(
        args.gen_model_name,
        device,
    )
    # Load the Filtering model.
    NLI_tokenizer, NLI_model = load_filtering_model(
        model_name=args.cls_model_name,
        device=device
    )
    NLI_model.eval()

    base_dir = Path(f"datasets/{args.task}")

    if args.gen_model_name not in ["t5-base", "t5-small", "t5-large", "t5-3b"]:
        checkpoint = args.gen_model_name.split("/")[-2]
        prefix = "entailment"
    else:
        checkpoint = args.gen_model_name
        prefix = "summarize"
    
    # E.g. t5-3b_huffpost_10N_top-k_40.txt
    save_dir = Path(f't5_generated/{checkpoint}_{args.task}_' + \
        f'{args.num_generate_per_sentence}N_' + \
        f'{args.decoding_strategy}_{decoding_value}')


    for i in range(args.num_exps):
        for mode in ['train', 'dev', 'test']:
            data_path = base_dir.joinpath(f"exp_{i}_10/{mode}.tsv")
            print(f"Start filtering {data_path}.")
            df = pd.read_csv(data_path, sep="\t", header=None)
            sentences = df.iloc[:, 1].tolist()
            sentences = prepare_data(sentences, prefix)
            generated_sentences = gen_t5_data(
                sentences=sentences,
                t5_tokenizer=t5_tokenizer,
                t5_model=t5_model,
                batch_size=args.batch_size,
                decoding_strategy=args.decoding_strategy,
                do_sample=do_sample,
                num_generate_per_sentence=args.num_generate_per_sentence,
            )
            # Start filtering sentences.
            pbar = trange(len(sentences))
            not_passed_counts = 0
            new_sentences = []
            for k in pbar:
                ori_sen_batch = [sentences[k]] * args.num_generate_per_sentence
                # Use list comprehension to append new_sen_batch.
                new_sen_batch = [generated_sentences[k * args.num_generate_per_sentence + i]
                                 for i in range(args.num_generate_per_sentence)]

                best_choice, got_passed = filter_sentences(
                    origin_sentence=ori_sen_batch,
                    new_sentence=new_sen_batch,
                    tokenizer=NLI_tokenizer,
                    model=NLI_model,
                    device=device,
                    filter_mode=args.filter_mode,
                )

                if not got_passed:
                    not_passed_counts += 1

                pbar.set_postfix(not_passed=not_passed_counts)

                if best_choice != -1:
                    best_sent = new_sen_batch[best_choice]
                    print(f"Ori sentence: {sentences[k]}")
                    print(f"Best sentence: {best_sent}")
                else:
                    best_sent = ""
                new_sentence = sentences[k] + " " + best_sent
                new_sentences.append(new_sentence)

            # save datasets
            output_dir = save_dir.joinpath(f"exp_{i}_10", "t5")
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)

            new_df = pd.DataFrame(
                {
                    "0": df.iloc[:, 0].tolist(),
                    "1": new_sentences,
                }
            )
            new_df.to_csv(
                f"{output_dir}/{mode}.tsv",
                sep="\t",
                header=None,
                index=False
            )

    with open(save_dir.joinpath('config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)