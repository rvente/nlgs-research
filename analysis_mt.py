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

pkl = list((root_path / "pipeline/predictions").glob("*mt*"))[1]
pkl
# %%
OUTPUT_PATH = root_path / "pipeline/scores" / pkl.name.removesuffix(".pkl")
OUTPUT_PATH.mkdir(exist_ok=True)
OUTPUT_PATH
# %% [markdown]
# ## First, Data to sentence.
# %%
preds_raw = pd.read_pickle(pkl)
test_predictions = preds_raw[preds_raw.task == 'd2s']
test_predictions 
# %%
compute_rouge = lambda x,y: rouge.compute(references=[x], predictions=[y], use_stemmer=False, use_aggregator=False)
compute_rouge(["general kenobi"], "general kenobi")
y_pred = test_predictions.drop(columns=['input_ids','attention_mask','pred_ids','labels'])
# %%

chunked = (
  seq(y_pred.to_dict('records'))
    .group_by(get.record_idx)
    .map(get[1]) # focus on the values
    .map(lambda x: [
      seq(x).map(get.sd).map(get[7:]).to_list(),        # gather up all of the references
      seq(x).map(get.decoded).to_list()[0][7:] # and the first prediction
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
scores_df = pd.concat(all_scores, axis=1)
scores_df
# %%
scores_df.describe()
# %%
scores_df.to_pickle(OUTPUT_PATH / "d2s_scores.pkl")
# %%
scores_df = pd.read_pickle(OUTPUT_PATH / "d2s_scores.pkl")
scores_df 
# %%
model_predictions = chunked.to_pandas()
model_predictions.columns = ['references','predictions']
joint_table = pd.concat([scores_df, model_predictions], axis=1)
worst_preds = joint_table.sort_values(by='bleu_bp').head(20)
worst_preds['palatul_count'] = worst_preds.predictions.map(lambda x: str(x.count("Palatul") ))
worst_preds['predictions'] = worst_preds.predictions.map(lambda x: x.replace("Palatul", "") )
pd.set_option('display.max_colwidth', None)
dspl_html(worst_preds[['predictions', 'palatul_count']]
            .applymap(lambda x: x[:240])
            # .to_html(index=False)
            .to_latex(index=False, multirow=True)
)
# %%
dspl_html(worst_preds[['references','predictions', 'palatul_count']]
            .applymap(lambda x: x[:240])
            .to_html(index=False)
)
# %%
test_predictions  = preds_raw[preds_raw.task == 's2d']
# %%
# define set notion of precision when multiple labels are assigned
# to a single instance, with epsilon preventing div by zero
def compute_f_measure(pred: set[str], gt: set[str], epsilon=1e-99):
    tp = len(pred.intersection(gt)) # pred true and actually true
    fp = len(gt - pred)             # in pred but not in gt
    fn = len(pred - gt)             # not in pred but actualy true

    prec = tp / (tp + fp + epsilon) 
    recl = tp / (tp + fn + epsilon) 
    f1 = (2 * prec * recl) / (prec + recl + epsilon)
    return f1

# %% [markdown]
# ## Unit Tests

# %% 
assert compute_f_measure(set("a"), set('a')) == 1
assert compute_f_measure(set("ab"), set('a')) == 2/3
assert compute_f_measure(set() , set('a')) == 0
# %%
test_predictions.decoded
# %%
# don't penalize for quotes or spaces
norm_split_set = lambda x: (
  x.str.upper()
  .str.replace("'", '')
  .str.replace(' ','')
  .str.replace("S2D\d:", "")
  .map(_.split(";")).map(set)
)
y_pred = norm_split_set(test_predictions.decoded)
y_pred
# %%
y_true = norm_split_set(test_predictions.sd)
y_true
# %%

f1_scores = seq(y_pred).zip(y_true).starmap(compute_f_measure).to_list()
f1_scores
# %%
def compute_closest_edit_dists(y_pred, y_true):
    # we need an alignment of the labels by edit distance
    return (
      seq(y_pred)
        .cartesian(y_true)
        .starmap(edit_distance) 
        .sorted() # full penalty for missed guesses or too many guesses
        .to_list()
    )

edit_distances = (
  seq(y_pred)
    .zip(y_true)
    .starmap(compute_closest_edit_dists)
    .map(np.mean)
    .to_list()
)
edit_distances 
# %%
results = test_predictions
results['f1_scores'] = f1_scores
results['med_scores'] = edit_distances # med mean edit distance
results 
# %%
results.describe()
# %%
# let's define nth finish and "place-number" as 0 for "finishing in first place"
# give find the place-number given a score: ties should have the same place
score_to_nth_finish: dict[float, int]= (
  seq(f1_scores)
    .sorted(reverse=True) # Highest to lowest
    .zip_with_index()     # gives an over-estimate of nth-finish
    .group_by(get[0])     # so we group by the f1 scores
    .map(get[1])          # then we get the actual place of the score
    .map(get[0])          # it's sorted, so take the first to account for ties
    .to_dict()            # convert to dictionary
)
seq(score_to_nth_finish.items()).to_pandas()
# %% [markdown]
#  so we can sort by this key later, and also get a broad impression
# of the distribution of errors. Later we'll plot a histogram anyway.

# %% [markdown]
# ## Error analysis
# %%
results['nth_finish'] = results['f1_scores'].map(score_to_nth_finish)
results['nth_finish'] 
# %%
results[['nth_finish','med_scores', 'f1_scores', 'decoded','sd']].round(3).to_csv('vis.csv')
# %%
# 
worst_finishes = results[results.f1_scores == 0]
dspl_html(worst_finishes[['med_scores','f1_scores', 'decoded','sd']].to_html())
print(len(worst_finishes))
# %%
results[['med_scores','f1_scores']].describe()
# %%
results.f1_scores.hist(bins=15)
# %%
worst_finishes.med_scores.hist(bins=15)
# %%
# sparse-bar formation of the same histogram data
ax = (
  worst_finishes
    .med_scores
    .sort_values()
    .map(lambda x: (x // 10) * 10)
    .map(lambda x: "[" + str(int(x)) + ", " + str(int(x+10)) + ")")
    .value_counts()
)
print(ax.to_latex())
print(ax.to_markdown())
# %%
worst_finishes.category.value_counts().plot.bar()
# %%
train_corpus = test_predictions
train_corpus.category.value_counts().plot.bar()
# %%
# normalized performance by category
npc = worst_finishes.category.value_counts() / train_corpus.category.value_counts()
npc.sort_values().plot.bar()
# %% [markdown]
# the network performs poorly on buildings, sports teams, and monuments when
# normalized for class prevalence. Poor performance on monument may be explained
# by its under-representation in the training set. This does not hold for sports
# teams and monuments, which have good representation in the training set but do
# not have good performance. This points to qualitative features particular to
# entries in those categories.
# %%
worst_finishes.sort_values(by=['med_scores'])
# %%