# Code for BART-GLC Training
Code for EACL 2023 paper: Enhancing Dialogue Summarization with Topic-Aware Global- and Local- Level Centrality.

> This is an incomplete version, the complete version will be updated after code cleaning.

The runing process:
1. download the pretrained model from https://huggingface.co/uer/bart-base-chinese-cluecorpussmall and placed on the `bart-glc/`;
2. `pip install -r requirements.txt`
3. `bash run_csds_glc.sh`

Some comments:
- We train our model on 4xV100 32G GPUs, you can change the training settings in `run_csds_glc.sh`.
- The results of our BART-GLC model are placed on the `results/`
- If you want to get the rouge, bleu, bertscore, and moverscore, you can move in the `eval_summ` dir.
- The core code of our paper is placed in the `src/models`.