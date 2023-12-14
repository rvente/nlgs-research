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
from sys import argv
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
dspl_html = lambda x: display_html(x, raw=True)
rouge = load('rouge')
print(argv)
# %%
index = int(argv[1]) if len(argv) == 2 else 0
root_path = Path("/home/vente/repos/nlgs-research")
pkl = (
  list( (root_path / "pipeline/predictions").glob("*s2d*")) +
  list( (root_path / "pipeline/predictions").glob("*mt*" ))
)[index]
print(pkl.name)
test_predictions = pd.read_pickle(pkl)
if 'task' in test_predictions.columns:
  test_predictions = test_predictions[test_predictions.task == 's2d']
test_predictions
# %%
OUTPUT_PATH = Path("/home/vente/repos/nlgs-research/pipeline/scores") / pkl.name.removesuffix(".pkl")
OUTPUT_PATH.mkdir(exist_ok=True)
OUTPUT_PATH
# %%
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
# FP <- PRED - GT |> length
# FN <- GT - PRED |> length  
# prec <- TP / (TP + FP)  
# recl <- TP / (TP + FN)  
# F1 <- harmonic_mean(prec, recl)  
# ```
# %%
# define set notion of precision when multiple labels are assigned
# to a single instance, with epsilon preventing div by zero
def compute_f_measure(pred: set[str], gt: set[str], epsilon=1e-99):
    tp = len(pred.intersection(gt)) # pred true and actually true
    fp = len(pred - gt)             # in pred but not in gt
    fn = len(gt - pred)             # not in pred but actualy true

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
# don't penalize for quotes or spaces
norm_split_set = lambda x: (
  x.str.upper()
   .str.replace("S2D.\\d:", '', regex=True)
   .str.replace("'", '')
   .str.replace(' ','')
   .map(_.split(";"))
   .map(set)
)
y_pred = norm_split_set(test_predictions['decoded'])
y_pred
# %%
y_true = norm_split_set(test_predictions['sd'])
y_true
# %%
f1_scores = seq(y_pred).zip(y_true).starmap(compute_f_measure).to_list()
seq(f1_scores).take(10)
# %%
def comptute_edit_distances(y_pred, y_true):
    # the arguments are sets of strings
    # to find the closest edit distances
    # take the cartesian product and limit
    # this works since y_pred and y_true do not contain duplicates
    # note that this penalizes for longer sequences
    return (
      seq(y_pred)
        .cartesian(y_true)
        .starmap(edit_distance) 
        .sorted()
        .to_list()
    )

edit_distances = (
  seq(y_pred)
    .zip(y_true)
    .starmap(comptute_edit_distances)
    .map(np.mean)
    .to_list()
)
seq(edit_distances).take(10)
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
# ## Error analysis
# %%
results['y_pred']  = y_pred
results['y_true']  = y_true
results['y_len']  = y_true.map(len)
results['nth_finish'] = results['f1_scores'].map(score_to_nth_finish)
results['nth_finish'] 
# %%
worst_finishes = results[results.f1_scores == 0]
dspl_html(worst_finishes[['med_scores','f1_scores', 'y_pred','y_true']].to_html())
print(len(worst_finishes))
# %%
results[['med_scores','f1_scores']].describe()
# %%
results.f1_scores.hist(bins=15)
plt.xlabel("$F_1$ Score")
plt.ylabel("Count")
plt.savefig(OUTPUT_PATH/'f1_hist.svg')
# %%
worst_finishes.med_scores.hist(bins=15)
plt.title("Edit Distances Among Predictions With $F_1$ score of 0.0")
plt.xlabel("Edit Distances")
plt.ylabel("Count")
plt.savefig(OUTPUT_PATH/'edit_dist.svg')
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
plt.savefig(OUTPUT_PATH/'worst_finishes_cats.svg')
# %%
# normalized performance by category
corpus = pd.read_pickle(root_path/'pipeline/normalized_data/webnlg_clean.pkl')
train_corpus = corpus[corpus.subset == 'train']
train_corpus.category.value_counts().plot.bar()
# %%
npc = worst_finishes.category.value_counts() / train_corpus.category.value_counts()
npc.sort_values().plot.bar()
plt.title("$F_1 = 0$ by Training Category")
plt.ylabel("Fraction of Training Samples")
plt.savefig(OUTPUT_PATH/'normalized_performance_by_cat.svg')
# %%
worst_finishes.sort_values(by=['med_scores'])
# %%
results[['med_scores','record_idx','f1_scores']].to_csv(OUTPUT_PATH/'results.csv')
y_true.to_csv(OUTPUT_PATH/'y_true.csv')
y_pred.to_csv(OUTPUT_PATH/'y_pred.csv')
worst_finishes.to_csv(OUTPUT_PATH/'worst_finishes.csv')
# %%
results.describe()
# %%
worst_finishes.describe()
# %%
