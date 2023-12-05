# %%
# this file takes in all files and produce the appropriate s2d analysis
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


INPUT_TO_ANALYSE = ""

# TODO: get this file form argv so driver code can run all analyses in a loop
pkl = max( (root_path / "pipeline/predictions").glob("*s2d*"))
pkl.name
# %%
OUTPUT_PATH = Path("/home/vente/repos/nlgs-research/pipeline/scores") / pkl.name.removesuffix(".pkl")
OUTPUT_PATH.mkdir(exist_ok=True)
OUTPUT_PATH
# %%
# corpus = pd.read_pickle(root_path / "pipeline/normalized_data/webnlg_clean.pkl")
# test_predictions = pd.read_pickle(pkl)
# test_corpus = corpus[corpus['subset']=='test'].copy()
# test_corpus 
# %%
compute_rouge = lambda x,y: rouge.compute(references=[x], predictions=[x], use_stemmer=False, use_aggregator=False, rouge_types=['rouge2'])

test_predictions['decoded']
# TODO: need to unflatten so interop with test corpus? Not quite.

# %%
test_predictions
# %% [markdown]
# How do we formulate F-measure for this task? Usually there is a fixed number
# of classes, and one label per class. But this class is fundimentally about
# extracting many labels from a single sequence.
# Approach: treat one data sample as multiple classification events, compute the
# f-measure across each label in an needs to be order-insensitive by comparing
# the sets themselves
# 
# ```
# TP <- PRED `intersect` GT |> length
# FP <- GT - PRED |> length
# FN <- PRED - GT |> length  
# F1 <- harmonic_mean(prec, recl)  
# prec <- TP / (TP + FP)  
# recl <- TP / (TP + FN)  
# ```
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
# %%
# don't penalize for quotes or spaces
norm_split_set = lambda x: x.str.upper().str.replace("'", '').str.replace(' ','').map(_.split(";")).map(set)
y_pred = norm_split_set(test_predictions['decoded'])
y_pred
# %%
y_true = norm_split_set(test_predictions['sd'])
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
        .sorted()
        # full penalty for missed guesses or too many guesses
        .take(seq(y_true, y_pred).map(len).max())
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
results['f1_scores'].hist(bins=15)
# %%
worst_finishes['med_scores'].hist(bins=15)
# %%
worst_finishes.category.value_counts().plot.bar()
# %%
train_corpus = corpus[corpus.subset == 'train']
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
