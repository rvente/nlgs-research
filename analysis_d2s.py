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

import matplotlib.pyplot as plt
# %%
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
root_path = Path("/home/vente/repos/nlgs-research")


# TODO: get this file form argv so driver code can run all analyses in a loop
# or just analayse the outputs of the best model...?
pkl = max( (root_path / "pipeline/predictions").glob("*d2s*"))
pkl.name
# %%
OUTPUT_PATH = root_path / "pipeline/scores" / pkl.name.removesuffix(".pkl")
OUTPUT_PATH.mkdir(exist_ok=True)
OUTPUT_PATH
# %%
test_predictions = pd.read_pickle(pkl)
# %%
compute_rouge = lambda x,y: rouge.compute(references=[x], predictions=[y], use_stemmer=False, use_aggregator=False)
compute_rouge(["general kenobi"], "general kenobi")
y_pred = test_predictions.drop(columns=['input_ids','attention_mask','pred_ids','labels'])

chunked = (
  seq(y_pred.to_dict('records'))
    .group_by(get.record_idx)
    .map(get[1]) # focus on teh values
    .map(lambda x: [
      seq(x).map(get.nl).to_list(),        # gather up all of the references
      seq(x).map(get.decoded).to_list()[0] # and the first prediction
    ])
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
compute_bleu = lambda x,y: bleu.compute(references=[x], predictions=[y])
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
scores_df = pd.concat(all_scores, axis=1)
scores_df
# %%
scores_df.describe()
# %%
scores_df.to_pickle(OUTPUT_PATH / "d2s_scores.pkl")
# %%
