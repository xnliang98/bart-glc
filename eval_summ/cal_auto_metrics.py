import numpy as np

import sacrebleu
def get_bleu(predictions,
        references,
        smooth_method="exp",
        smooth_value=None,
        force=False,
        lowercase=False,
        tokenize=None,
        use_effective_order=False):
    references_per_prediction = len(references[0])
    if any(len(refs) != references_per_prediction for refs in references):
        raise ValueError("Sacrebleu requires the same number of references for each prediction")
    transformed_references = [[refs[i] for refs in references] for i in range(references_per_prediction)]
    # print(transformed_references)
    output = sacrebleu.corpus_bleu(
        predictions,
        transformed_references,
        smooth_method=smooth_method,
        smooth_value=smooth_value,
        force=force,
        lowercase=lowercase,
        use_effective_order=use_effective_order,
        **(dict(tokenize=tokenize) if tokenize else {}),
    )
    output_dict = {
        "score": output.score,
        "counts": output.counts,
        "totals": output.totals,
        "precisions": output.precisions,
        "bp": output.bp,
        "sys_len": output.sys_len,
        "ref_len": output.ref_len,
    }
    # return output_dict
    return "{:.2f}".format(output_dict['score'])

from bert_score import score
def get_bertscore(cands, refs, lang="en"):
    if lang == "en":
        (P, R, F), hashname = score(cands, refs, model_type="roberta-large", return_hash=True)
    elif lang == "zh":
        (P, R, F), hashname = score(cands, refs, model_type="bert-base-chinese", return_hash=True)
    res = f"{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={F.mean().item():.6f}"
    # return res
    return "{:.2f}".format(F.mean().item()*100)

from moverscore_v2 import get_idf_dict, word_mover_score
def get_moverscore(preds, refs):
    idf_dict_hyp = get_idf_dict(preds)
    idf_dict_ref = get_idf_dict(refs)
    scores = word_mover_score(refs, preds, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=2, batch_size=32)
    # mover_avg_score = np.mean()
    mover_avg_score = np.array(scores).mean().item()
    return  "{:.2f}".format(mover_avg_score*100)

def load_lines(path, sep=None):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if sep == "":
        lines = [" ".join(list(line.strip())) for line in lines]
    else:
        lines = [line.strip() for line in lines]

    return lines

def convert_to_ids(pred, ref):
    ref_id, pred_id = [], []
    tmp_dict = {'%': 0}
    new_index = 1
    words = list(ref)
    for w in words:
        if w not in tmp_dict.keys():
            tmp_dict[w] = new_index
            ref_id.append(str(new_index))
            new_index += 1
        else:
            ref_id.append(str(tmp_dict[w]))
        if w == '。':
            ref_id.append(str(0))
            ref_id.append("\n")
    words = list(pred)
    for w in words:
        if w not in tmp_dict.keys():
            tmp_dict[w] = new_index
            pred_id.append(str(new_index))
            new_index += 1
        else:
            pred_id.append(str(tmp_dict[w]))
        if w == '。':
            pred_id.append(str(0))
            pred_id.append("\n")
    return ' '.join(ref_id), ' '.join(pred_id)

from rouge_score import rouge_scorer, scoring
def get_rouge(preds, refs, lang="en", rouge_types=None, use_stemmer=False):
    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeLsum"]

    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()
    for ref, pred in zip(refs, preds):
        if lang == 'zh':
            ref, pred = convert_to_ids(pred, ref)
        score = scorer.score(ref, pred)
        aggregator.add_scores(score)  
    result = aggregator.aggregate()
    res_dict = {}
    for key in rouge_types:
        res_dict[key] = result[key].mid.fmeasure

    # return res_dict
    return "{:.2f}".format(res_dict['rouge1']*100), "{:.2f}".format(res_dict['rouge2']*100),  "{:.2f}".format(res_dict['rougeLsum']*100)




cands = load_lines("/root/projects/my-bart/results/bart_csds_user.preds")
refs = load_lines("../bart/results/csds_user.refs")


print(cands[0])
print(refs[0])

cands_lst = [cands]
refs_lst = [refs]

bleu_scores = []
r1_scores = []
r2_scores = []
rL_scores = []
bs_scores = []
ms_scores = []

for cands, refs in zip(cands_lst, refs_lst):
    bleu_scores.append(str(get_bleu([" ".join(list(c)) for c in cands], [[" ".join(list(r))] for r in refs])))
    r1, r2, rL = get_rouge(cands, refs, lang="zh")
    r1_scores.append(str(r1))
    r2_scores.append(str(r2))
    rL_scores.append(str(rL))

    bs_scores.append(get_bertscore(cands, refs, "zh"))
    ms_scores.append(get_moverscore(cands, refs))

print(f'BLEU:{"/".join(bleu_scores)}\nR1:{"/".join(r1_scores)}\nR2:{"/".join(r2_scores)}\nRL:{"/".join(rL_scores)}\nBS:{"/".join(bs_scores)}\nMS:{"/".join(ms_scores)}')


