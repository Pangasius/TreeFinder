## Steps to retrieve plots in the paper

These are the plots exploring the relationship between corroborative attributions and contributive attributions.

### Download Loogle datasets

- Download longdep_qa.json
- Download shortdep_qa.json

### Compute logprobs for each chunk

```bash
 python Prediction/chart_vllm.py --task longdep_qa --max_length 32000 --factor 4
```
```bash
python Prediction/chart_vllm.py --task shortdep_qa --max_length 32000 --factor 4
```

### Clean format and plot

Do one at a time or they will overwrite:

- Longdep

```bash
python Prediction/analyze.py --task longdep_qa --file_path Output/longdep_qa_Qwen.Qwen2.5-7B-Instruct-1M_CHART_8000.jsonl --factor 1
```
```bash
python Prediction/plot.py --factor 1
```

- Shortdep

```bash
python Prediction/analyze.py --task shortdep_qa --file_path Output/shortdep_qa_Qwen.Qwen2.5-7B-Instruct-1M_CHART_8000.jsonl --factor 1
```
```bash
python Prediction/plot.py --factor 1
```
