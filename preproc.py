
# %%
# %%
from datasets import load_dataset, Dataset
from evaluate import load
from operator import itemgetter
import operator as op
import re

# %%
import datasets
import random
import functools
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from collections import Counter
from IPython.display import display, HTML
# %%
from functional import seq
from funcutils import underscore as _
from funcutils import get 
from unidecode import unidecode
# %%
# import webnlg 2.0
raw_datasets = load_dataset("web_nlg", "release_v2")
raw_datasets

df_raw = pd.concat([
        pd.concat([
            pd.DataFrame(raw_datasets[e]),
            pd.DataFrame([e] * len(raw_datasets[e]), columns=['subset'])
        ], axis=1)
        for e in ['dev','train','test'] 
    ],
    axis=0)

df_raw = df_raw.reset_index()
df = df_raw[['subset','category','index']]
df
# %% 

# natural language
nl_len = df_raw['lex'].map(get["text"]).map(len)
nl_len
# %%
# TODO: find sentences that repeat the same content in multiple ways.
nl = df_raw['lex'].map(get['text']).map(lambda x: " ".join(x))
nl
# %%
# structured data
sd = df_raw['modified_triple_sets'].map(get['mtriple_set']).map(get[0])
sdl = sd.map(len)
display(sd)
display(sdl)
# %%
# normalized structured data

def normalize_terms(rdf_triples: list[str]):
    '''surround terms, remove _ and " as well '''
    camelcase = re.compile(r'(?<!^)(?=[A-Z])')

    # camelCase to space separated, except for all-caps words
    de_camelcase = lambda x: camelcase.sub(' ', x).lower() if not x.upper() == x else x
    decamelcase_middle = lambda x: [x[0], de_camelcase(x[1]), x[2]]

    def join_with_bar(triple: list[str]):
       return seq(triple).reduce(lambda x,y: x + "|" + y)

        
    return (
        seq(rdf_triples)
          .map(_.replace("_", " "))
          .map(_.replace('"', ""))
          .map(_.replace(';', "")) # only 40 of these exist
          .map(_.split(" | "))
          .map(decamelcase_middle)
          .map(join_with_bar)
    )


nsd = sd.map(normalize_terms).map(lambda x: "; ".join(x)).map(unidecode)
nsd
# %%
# %% [markdown]
# Ensure that the operations just performed are invertable by reformatting them as triples.
# While we're at it, perform some data cleaning
# including normalizing to unicode
# %%
vocab_freq = seq(nsd).map(Counter).reduce(op.add)
seq(vocab_freq.keys()).sorted().reduce(op.add)
# %%
len(vocab_freq.keys())
# %% [markdown]
# Now the actual checking: let's ensure that all elements are well-formed with 3 terms in each label
# %%
counts = nsd.map(_.split(";")).map(lambda x: seq(x).map(_.split('|')).map(len)).map(Counter)
[(trmlen, freq)] =  seq(counts.values).reduce(op.add).most_common()
assert trmlen == 3
assert freq >= len(counts.values)
# %%
df['sd'] = nsd
df
# %%
# normalize by removing ascii
nnl = nl.map(unidecode)
df['nl'] = nnl
# %%
df
# %%
seq(df.loc[16093].to_dict().items())
# %%
df.to_pickle("~/repos/nlgs-research/pipeline/normalized_data/webnlg_clean.pkl")
# %%