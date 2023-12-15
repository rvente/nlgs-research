# %%
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
from transformers import AutoTokenizer
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
df = pd.read_pickle('pipeline/normalized_data/webnlg_raw.pkl')


df['lex_char_len'] = df.nl.map(lambda x: " ".join(x)).str.len()
df['rdf_char_len'] = df.sd.map(lambda x: " ".join(x)).str.len()
# %%
df.describe().round(2)
# %%
print(
    df.category.value_counts().to_latex()
)
# %%
label_dist = df.sd.map(lambda x: ";".join(x)).map(_.split("|")).map(get[1])
label_dist.value_counts().hist()
label_dist.value_counts()
# %%
counts_alone = label_dist.value_counts().values
# %%

# Function x**(1/2)
def forward(x):
    return x**(1/2)


def inverse(x):
    return x**2
ax = plt.hist(counts_alone, bins=100, density=False)
plt.xlabel('Label Occurrences')
plt.ylabel('Frequency')
plt.yscale('log')
# %%
# %%
wb = pd.read_pickle("~/repos/nlgs-research/pipeline/normalized_data/wikibio.pkl").sample(8000)
wb['char_len'] = wb.target_text.str.len()
wb.target_text = (
    wb.target_text
    .str.replace("-lrb- ", "(")
    .str.replace(" -rrb-", ")")
    .str.replace(" ,", ",")
)
wb.target_text 
# %%
wb.describe().round(2)

print(wb[wb.char_len > 100][wb.char_len < 500].iloc[-1].target_text)
wb[wb.char_len > 100][wb.char_len < 500]
# %%
from evaluate import load

bertscore = load('bertscore')
compute_bert = lambda x,y: bertscore.compute(predictions=[y], references=[x], lang="en", model_type="distilbert-base-uncased" )

# %%
preds = pd.read_pickle("~/repos/nlgs-research/pipeline/predictions/mt-t5-base-5.pkl")
scores = pd.read_pickle("~/repos/nlgs-research/pipeline/scores/mt-t5-base-5/d2s_scores.pkl")
scores
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

X = scores.bert_f1
Y = scores.bleu_score
Z = scores.bleu_sys_len


fig = plt.figure()

cmap = cm.get_cmap('coolwarm')
color = cmap(Z)[..., :3]

ax = plt.scatter(X,Y,c=color)
plt.xlabel('BERTScore')
plt.ylabel('BLEU')
from scipy.stats.stats import pearsonr   
pearsonr(X, Y)
# %%
ax = plt.scatter(X,Z,c=color)
plt.xlabel('BERTScore')
plt.ylabel('Length')
pearsonr(X,Z)

# %%
preds[preds.category == 'SportsTeam'][preds.task =='d2s']
# %%
