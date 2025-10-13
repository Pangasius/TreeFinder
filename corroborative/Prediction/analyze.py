import argparse
from datasets import Dataset
from fuzzysearch import find_near_matches
from tqdm import tqdm
import pandas as pd


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path", type=str, default=None, help="file path to the dataset"
    )
    parser.add_argument(
        "--factor", type=int, default=1, help="factor to split the input"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="long context understanding tasks in LooGLE",
        choices=[
            "shortdep_qa",
            "longdep_qa",
            "longdep_summarization",
            "shortdep_cloze",
        ],
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args()

    data = pd.read_json(path_or_buf="data/" + args.task + ".json", lines=True)
    # convert to dataset
    data = Dataset.from_pandas(data)

    # verify name
    if args.task not in args.file_path:
        print("Error: Task name mismatch")
        exit(1)

    if str(8000 // args.factor) not in args.file_path:
        print("Error: Factor mismatch")
        exit(1)

    with open(args.file_path, "r") as f:
        output = pd.read_json(path_or_buf=f, lines=True)

    with open(str(8000 // args.factor) + "_output.txt", "w") as f:
        headers = [
            "TextNumber",
            "QuestionNumber",
            "DeletedNumber",
            "ContainedHint",
            "logprob",
            "Sublogprob",
        ]
        f.write("\t".join(headers) + "\n")

    output_index = 0
    for i, data_instance in enumerate(tqdm(data)):
        raw_inputs = data_instance["input"]

        if (
            eval(data_instance["qa_pairs"])[0]["Q"]
            != output["qa_pairs"][output_index][0]["Q"]
        ):
            print("Error: Mismatched Q")
            continue

        splits = raw_inputs.split(".")

        # Group splits
        splits_grouped = []
        current_group = ""
        for split in splits:
            current_group += split + "."
            if len(current_group) + len(split) > 8000 // args.factor:
                splits_grouped.append(current_group)
                current_group = ""

        logprobs = output["logprobs"][output_index]

        sub_logprobs = output["logprobs_sub"][output_index]

        if len(logprobs) != len(eval(data_instance["qa_pairs"])):
            print(
                "Error: Mismatched Length: ",
                len(logprobs),
                len(eval(data_instance["qa_pairs"])),
            )
            output_index += 1
            continue

        for j, pairs in enumerate(eval(data_instance["qa_pairs"])):
            hint_sentences = []

            if type(pairs["S"]) is str:
                pairs["S"] = [pairs["S"]]

            not_found = 0
            for s in pairs["S"]:
                first_attempt = True
                found = 0
                sr = s[:50]
                while found > -1:
                    found += 1
                    if first_attempt and found > 10:
                        sr = s[-50:]
                        found = 0
                        first_attempt = False

                    if found > 10:
                        break
                    for k, split in enumerate(splits_grouped):
                        result = find_near_matches(sr, split, max_l_dist=found)
                        if len(result) > 0:
                            found = -1
                            hint_sentences += [k]
                            break
                if found > 0:
                    not_found += 1
                    hint_sentences += [-1]

            if not_found == len(pairs["S"]):
                print("Error: Not Found")
                output_index += 1
                continue

            number_erased = {k: 0 for k in range(len(splits_grouped))}
            for k in range(len(splits_grouped)):
                for h in hint_sentences:
                    if h == k:
                        number_erased[k] += 1

            if not_found == 0 and len(hint_sentences) > 0:
                # mean sub_logprob diff of hinted vs non-hinted
                total_confounded = sum(sub_logprobs[j])
                different_hinted = [
                    sub_logprobs[j][k] for k in hint_sentences if k != -1
                ]
                total_hinted = sum(different_hinted)
                total_non_hinted = total_confounded - total_hinted

                mean_confounded = total_confounded / len(sub_logprobs[j])
                mean_hinted = total_hinted / len(different_hinted)
                mean_non_hinted = (
                    0
                    if (len(sub_logprobs[j]) - len(different_hinted)) == 0
                    else total_non_hinted
                    / (len(sub_logprobs[j]) - len(different_hinted))
                )

            for k in range(len(splits_grouped)):
                with open(str(8000 // args.factor) + "_output.txt", "a") as f:
                    f.write(
                        "\t".join(
                            [
                                str(i),
                                str(j),
                                str(k),
                                str(k in hint_sentences),
                                str(logprobs[j]),
                                str(sub_logprobs[j][k]),
                            ]
                        )
                        + "\n"
                    )

        output_index += 1
        print(len(output["qa_pairs"]))
        if output_index >= len(output["qa_pairs"]):
            break
