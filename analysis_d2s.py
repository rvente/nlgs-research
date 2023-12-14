# %%
# this file takes in all files and produce the appropriate d2s analysis
from pathlib import Path
from datasets import load_dataset
from evaluate import load
import pandas as pd
import numpy as np
from pathlib import Path
from functional import seq
from funcutils import underscore as _
from funcutils import get
from IPython.display import display, display_html, HTML
from editdistance import distance as edit_distance
from sys import argv

import matplotlib.pyplot as plt
# %%
# I prefer these LaTeX plots to fit in with the paper better
plt.style.use('seaborn-v0_8-whitegrid')
params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
          "text.usetex" : True,
          "font.family" : "serif",
          "font.serif" : ["Computer Modern Serif"]}
plt.rcParams.update(params)

# %%
dspl_html = lambda x: display_html(x, raw=True)
rouge = load('rouge')
# %%
index = int(argv[1]) if len(argv) == 2 and argv[1].isnumeric() else 0
print(index)

root_path = Path("/home/vente/repos/nlgs-research")
pkl = (
  list( (root_path / "pipeline/predictions").glob("*d2s*")) +
  list( (root_path / "pipeline/predictions").glob("*mt*" ))
)[index]
print(pkl.name)
# %%
OUTPUT_PATH = root_path / "pipeline/scores" / pkl.name.removesuffix(".pkl")
OUTPUT_PATH.mkdir(exist_ok=True)
OUTPUT_PATH
# %%
test_predictions = pd.read_pickle(pkl)
is_mt = False
if 'mt' in pkl.name:
  test_predictions = test_predictions[test_predictions.task == 'd2s']
  is_mt = True
test_predictions
# %%
compute_rouge = lambda x,y: rouge.compute(references=[x], predictions=[y], use_stemmer=False, use_aggregator=False)
compute_rouge(["general kenobi"], "general kenobi")
y_pred = (
  test_predictions.drop(columns=['input_ids','attention_mask','pred_ids','labels'])
)

# we trained record by record but are evalating by test index,
# so chunk up the indices as appropriate

def conditional_cleaning(x):
  if is_mt:
    lead_trim = 7 # remove "d2s 0:" string identifier from start
    return [
      seq(x).map(get.sd).map(lambda x: x[lead_trim:]).to_list(),
      seq(x).map(get.decoded).to_list()[0][lead_trim:]
    ]
  else:
    return [
      seq(x).map(get.nl).to_list(),        # gather up all of the references
      seq(x).map(get.decoded).to_list()[0] # and the first prediction
    ]

chunked = (
  seq(y_pred.to_dict('records'))
    .group_by(get.record_idx)
    .map(get[1]) # focus on teh values
    .map(conditional_cleaning)
)
chunked
# %%
rouge_scores = (
  chunked.starmap(compute_rouge)
    # only one rouge per sample, so take the first of the values
    .map(lambda x: seq(x.items()).starmap(lambda x,y : {x:y[0]}))
    # rolling union on dictionaries since they are disjoint
    .map(lambda x: seq(x).reduce(lambda x, y: x | y))
    .to_pandas()
)
rouge_scores
# %%
rouge_scores.describe()
# %%
bleu = load('sacrebleu')
bleu
compute_bleu = lambda x,y: bleu.compute(references=[x], predictions=[y],lowercase=True, tokenize='intl')
# %%
bleu_scores = (
  chunked.starmap(compute_bleu)
    .to_pandas()
)
# %%
bleu_scores
# %%
bertscore = load('bertscore')
compute_bert = lambda x,y: bertscore.compute(predictions=[y], references=[x], lang="en", model_type="distilbert-base-uncased" )
# %%

bert_scores = (
 chunked
   .starmap(compute_bert)
   .to_pandas()
   .drop(columns='hashcode')
   .applymap(np.mean)
)
# %%
prepend_name_to_cols = lambda x,y : x.rename(columns=lambda e: y+"_"+e)
prepend_name_to_cols(bert_scores, 'bert')
all_scores = (
  seq(bert_scores, bleu_scores, rouge_scores)
    .zip(['bert','bleu','rouge'])
    .starmap(prepend_name_to_cols)
)
unflt = pd.DataFrame(chunked, columns=['references','predictions'])
scores_df = pd.concat(all_scores, axis=1)
scores_df
# %%
scores_preds = pd.concat([scores_df,unflt], axis=1)
scores_preds 
# %%
scores_df.describe()
# %%
scores_df.to_csv(OUTPUT_PATH / "d2s_scores.csv")
scores_df.to_pickle(OUTPUT_PATH / "d2s_scores.pkl")
# %%
scores_df.sort_values(by='bleu_score')
# %%
scores_preds.sort_values(by='bleu_score')

# %%
zero_bleus = scores_preds[scores_preds.bleu_score == 0]
zero_bleus
# %%
zero_bleus.shape
# %%
scores_preds.bleu_score.hist()
plt.title("Distribution of BLEU Scores")
plt.xlabel("BLEU Scores")
plt.ylabel("Count")
plt.savefig(OUTPUT_PATH/'bleu_score_dist.svg')
# %%
plt.clf()
# %%
scores_preds.bert_f1.hist()
plt.title("Distribution of BERTScores")
plt.xlabel("BERTScore")
plt.ylabel("Count")
plt.savefig(OUTPUT_PATH/'bertscore_dist.svg')
# %%
plt.clf()
# %%
scores_preds.rouge_rougeL.hist()
plt.title("Distribution of RougeL Scores")
plt.xlabel("Rouge LCS Score")
plt.ylabel("Count")
plt.savefig(OUTPUT_PATH/'rouge_dist.svg')
# %%
plt.clf()
# %%
