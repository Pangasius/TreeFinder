import os
import time
import torch
import json
import argparse
from transformers import AutoTokenizer
from openai import OpenAI
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
import concurrent.futures


URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct-1M",
        help="raw model name for evaluation",
        choices=[
            "Qwen/Qwen2.5-7B-Instruct-1M",
        ],
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="long context understanding tasks in LooGLE",
        choices=[
            "shortdep_qa",
            "longdep_qa",
        ],
    )
    parser.add_argument(
        "--max_length", type=int, default=None, help="the max length of input prompt"
    )

    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="./Output/")

    parser.add_argument(
        "--factor", type=int, default=1, help="factor to split the input"
    )

    return parser.parse_args(args)


def call_one_prompt(
    prompt, client, model_name, max_gen, prefilled_text, idx=0
):
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "The answer is: "},
    ]

    rsp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        top_p=1,
        max_tokens=max_gen,
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=1,
        extra_body={
            "guided_choice": None if prefilled_text == "" else [prefilled_text],
            "continue_final_message": True,
            "add_generation_prompt": False,
            "guided_decoding_backend": "outlines"
        },
    )

    log_probs = 0
    for logits in rsp.choices[0].logprobs.content:
        log_probs += logits.logprob

    return (
        rsp.choices[0].message.content,
        log_probs,
        0 if prefilled_text == "" else len(rsp.choices[0].logprobs.content),
        idx
    )


def call_model(
    json_obj,
    client,
    model_name,
    tokenizer,
    max_length,
    max_gen,
    prompt_format,
    factor,
):
    prompt = prompt_format.format(**json_obj)
    tokenized_prompt = tokenizer(
        prompt, truncation=False, return_tensors="pt"
    ).input_ids[0]

    if len(tokenized_prompt) > max_length or len(json_obj["input"]) < 2048:
        return None, None, None, None, None

    # First, divide the prompt into each sentence
    splits = json_obj["input"].split(".")

    # Group splits
    splits_grouped = []
    current_group = ""
    for split in splits:
        current_group += split + "."
        if len(current_group) + len(split) > 8000 // factor:
            splits_grouped.append(current_group)
            current_group = ""

    splits_numbers = [i for i in range(len(splits_grouped))]
    forbidden_numbers: list[int] = []
    print(
        "Found {} groups in the prompt of size {}.".format(
            len(splits_numbers), len(tokenized_prompt)
        ),
        flush=True,
    )

    prompt = prompt_format.format(**json_obj)
    og_text, _, total_length, _ = call_one_prompt(
        prompt, client, model_name, max_gen, ""
    )
    # Get comparison logprob
    _, og_logprob, _, _ = call_one_prompt(
        prompt, client, model_name, max_gen, og_text, idx=-1
    )

    # Compute sub-logprobs
    texts = []
    # Compute sub-logprobs
    logprobs = torch.zeros(len(splits_numbers))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for new_forbidden in set(splits_numbers) - set(forbidden_numbers):
            # Exclude the new forbidden number
            forbidden_numbers_local = forbidden_numbers + [new_forbidden]
            splits_chosen = set(splits_numbers) - set(forbidden_numbers_local)

            splits_chosen_ordered_list = sorted(list(splits_chosen))

            # Construct the new input
            prompt_args = json_obj.copy()
            prompt_args["input"] = [
                splits_grouped[i] for i in splits_chosen_ordered_list
            ]
            prompt_args["input"] = "".join(prompt_args["input"])

            prompt = prompt_format.format(**prompt_args)
            future = executor.submit(
                call_one_prompt,
                prompt,
                client,
                model_name,
                max_gen,
                og_text,
                new_forbidden,
            )
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            _, logprob, _, idx = future.result()
            logprobs[idx] = logprob

            print(
                "Current_logprob: ", logprob, " vs. Original_logprob: ", og_logprob
            )

        print("Current_logprob: ", logprob, " vs. Original_logprob: ", og_logprob)

    return og_text, og_logprob, total_length, texts, logprobs.tolist()


def get_pred(
    client, model_name, data_instance, tokenizer, max_length, max_gen, prompt_format
):
    ans, groundtruth, logprobs = [], [], []
    ans_sub, logprobs_sub = [], []
    preds = {}
    raw_inputs = data_instance["input"]

    total_length = 0
    preds["qa_pairs"] = eval(data_instance["qa_pairs"])
    for j in eval(data_instance["qa_pairs"]):
        json_obj = {"Q": j["Q"], "input": raw_inputs}

        pred, logprob, length, pred_new, logprobs_new = call_model(
            json_obj,
            client,
            model_name,
            tokenizer,
            max_length,
            max_gen,
            prompt_format,
            factor=args.factor,
        )

        if pred is None:
            continue

        ans.append(pred)
        groundtruth.append(j["A"])
        logprobs.append(logprob)
        ans_sub.append(pred_new)
        logprobs_sub.append(logprobs_new)
        total_length += length

    preds["llm_output"] = ans
    preds["output"] = groundtruth
    preds["logprobs"] = logprobs
    preds["llm_output_sub"] = ans_sub
    preds["logprobs_sub"] = logprobs_sub

    return preds, total_length


if __name__ == "__main__":
    args = parse_args()

    # data = load_dataset("bigainlco/LooGLE", args.task, split="test")
    data = pd.read_json(path_or_buf="data/" + args.task + ".json", lines=True)
    # convert to dataset
    data = Dataset.from_pandas(data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path + args.model_name)

    client = OpenAI(base_url=URL, api_key=API_KEY, timeout=None)

    tokenizer.pad_token = tokenizer.eos_token

    task2prompt = json.load(open("./config/task2prompt.json", "r"))
    task2maxlen = json.load(open("./config/task2maxlen.json", "r"))
    prompt_format = task2prompt[args.task]
    max_gen = task2maxlen[args.task]

    save_path = (
        args.output_path
        + args.task
        + "_"
        + args.model_name.replace("/", ".")
        + "_CHART_"
        + str(8000 // args.factor)
        + ".jsonl"
    )

    # If we resume from a previous run, we need to count what has been done
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            output = pd.read_json(path_or_buf=f, lines=True)
    else:
        output = []

    total_tokens = 0
    start_time = time.time()
    output_index = 0
    for i, data_instance in enumerate(tqdm(data)):
        if output_index < len(output):
            if (
                eval(data_instance["qa_pairs"])[0]["Q"]
                == output["qa_pairs"][output_index][0]["Q"]
            ):
                output_index += 1
            continue

        print("Processing instance ", i, " with output index ", output_index)

        preds, length = get_pred(
            client,
            args.model_name,
            data_instance,
            tokenizer,
            args.max_length,
            max_gen,
            prompt_format,
        )

        if preds is None:
            continue

        total_tokens += length
        elapsed_time = time.time() - start_time
        token_per_sec = total_tokens / elapsed_time
        tqdm.write(f"Token per second: {token_per_sec:.2f}")

        with open(save_path, "a+") as g:
            g.write(json.dumps(preds) + "\n")
