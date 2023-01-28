# requirements
- bert-score
- rouge-score
- sacrebleu
- numpy
- transformers
- torch
- pyemd

# required pre-trained models
- https://huggingface.co/hfl/chinese-bert-wwm for moverscore
- https://huggingface.co/bert-base-chinese for bert-score

# How to run
change the file path in `cal_auto_metric.py`. Then, `python cal_auto_metric.py`. 