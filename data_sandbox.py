# %% [markdown]
# If you're opening this Notebook on colab, you will probably need to install 🤗 Transformers and 🤗 Datasets as well as other dependencies. Uncomment the following cell and run it.

# %%
from huggingface_hub import notebook_login

# %%
# %%
from datasets import load_dataset
from evaluate import load

raw_datasets = load_dataset("web_nlg", "release_v2")
metric = load("rouge")
raw_datasets
# %%
raw_datasets["train"][0]

# %%
import datasets
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
# %%
%config InlineBackend.figure_format='svg'
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
df = pd.DataFrame({'lab':['A', 'B', 'C'], 'val':[10, 30, 20]})
ax = df.plot.bar(x='lab', y='val', rot=0)
# %%

import random
from IPython.display import display, HTML


# %%
df = pd.concat([
    pd.concat([pd.DataFrame(raw_datasets[e]), pd.DataFrame([e] * len(raw_datasets[e]), columns=['subset'])], axis=1) for e in ['dev','train','test']], axis=0)
display(df)
# %%
bar = pd.DataFrame(df["category"].value_counts()).plot.bar()
display(bar)
# %%
# %%
from transformers import AutoTokenizer

model_checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# %%
# tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
# %%

test_predictions = pd.read_pickle("metrics/t5-small-s2d-test.pkl")
test_predictions["predicted"] = test_predictions['predicted'].str.removesuffix("</s>")

# %%
tdf = pd.DataFrame(raw_datasets['test'])
# %%
rouge = load('rouge')
# %%
# test_predictions['rouge'] = (
#     (test_predictions['predicted'].map(lambda x: [x]) + actual)
#     .map(lambda x: rouge.compute(references=[x[1]], predictions=[x[0]], use_stemmer=False, use_aggregator=False, rouge_types=['rouge2']))
#     .map(lambda x: x['rouge2'][0])
# )
# %%
# paired = pd.DataFrame([test_predictions['predicted'].map(lambda x: x.split(". ")), actual]).T # d2s
# actual = tdf['lex'].map(lambda x: x['text'])
actual = tdf['modified_triple_sets'].map(lambda x: x['mtriple_set'][0]).map(lambda x: " ".join(x))
actual
# %%
paired = pd.DataFrame([test_predictions['predicted'].map(lambda x: x.split(". ")), actual]).T # s2d
paired
# %%
actual
# %%
# rouges = paired.apply(lambda x: rouge.compute(references=[x['lex']]*(len(x['predicted'])), predictions=x['predicted'],  rouge_types=['rougeL'],   use_aggregator=False), axis=1).map(lambda x: x['rougeL']).map(np.mean) # data to sentence
rouges = paired.apply(lambda x: rouge.compute(references=[x['modified_triple_sets']]*(len(x['predicted'])), predictions=x['predicted'],  rouge_types=['rougeL'],   use_aggregator=False), axis=1).map(lambda x: x['rougeL']).map(np.mean)
test_predictions['rouge'] = rouges
# %%

rouges.mean()
# %%
test_predictions.sort_values(by='rouge').head()[['summary', 'predicted','rouge']]
# %%
bleu = load('bleu')
# %%
bleus = paired.apply(lambda x: bleu.compute(references=[x['lex']]*(len(list(set(x['predicted']))[:4])), predictions=list(set(x['predicted']))[:4]), axis=1)#.map(lambda x: x['bleu']).map(np.mean)
bleus 
bleus.to_pickle("metrics/bleus-t5-base-bleus-d2s.pkl")
# %%
# TODO  find scores with maximum rouge alignment ahead of computing BERTScore?
bleus = pd.read_pickle("metrics/bleus-t5-base-bleus-d2s.pkl").map(lambda x: x['bleu']).map(np.mean)
bleus.mean()
# %%
test_predictions['bleu'] = bleus
# %%
test_predictions['acc'] = (
    (test_predictions['predicted'] == test_predictions['summary']).map(int)
)

# %%
test_predictions['acc'] .describe()
# %%
import re

input_text = '"AEK Athens F.C." | numberOfMembers | 69618 ""AEK Athens F.C."" | ground | Greece AEK_Athens_F.C. | league | Superleague_Greece AEK_Athens_F.C. | season | 2014 AEK_Athens_F.C. | numberOfMembers | 69618 AEK_Athens_F.C. | ground | Greece AEK_Athens_F.C. | season | 2014 AEK_Athens_F.C. | league | Superleague_Greece '

# Use regex to find text within double quotes and replace spaces with underscores
normalize = lambda x: re.sub(r'"([^"]*)"', lambda match: match.group(1).replace(' ', '_'), x.replace('""','"'))

print(normalize(input_text))
# %%

def parse_string_into_triples(input_string):
    # Split the input string into individual terms
    terms = input_string.split(' | ')

    # Create a list to store triples
    triples = []

    # Iterate through the terms and create triples
    for i in range(0, len(terms), 3):
        triple = " ".join(terms[i:i+3])
        triples.append(triple)

    return triples

input_string = 'Mid-Atlantic_Regional_Spaceport_Launch_Pad_0 | associatedRocket | Minotaur_IV Antares_(rocket) | comparable | Delta_II Delta_II | countryOrigin | United_States Antares_(rocket) | launchSite | Mid-Atlantic_Regional_Spaceport_Launch_Pad_0'
result = parse_string_into_triples(input_string)
print(result)
# %% tdf
test_predictions['normed'] = test_predictions['predicted'].map(normalize).map(parse_string_into_triples).to_csv('vis.csv',sep="\t")

test_predictions[['predicted','normed','rouge']].sort_values(by=['rouge']).to_csv('vis.csv', sep="\t"
                        )
# %%

test_predictions.describe()

test_predictions
# %%
test_predictions['sentences'] = test_predictions["lex"].map(lambda x: " ".join(x['text']))
test_predictions['num_sentences'] = test_predictions["lex"].map(lambda x: len(x['text']))
test_predictions['sentencelen'] = test_predictions["sentences"].map(len)
test_predictions['sentences']
# %%
# test_predictions.plot.scatter(x='sentencelen', y='rouge')

# Create a scatter plot using seaborn
# sns.scatterplot(x='sentencelen', y='rouge', data=test_predictions, s=30)

# Add a line of best fit
sns.regplot(x='sentencelen', y='rouge', data=test_predictions, ci=None, line_kws={'color': 'red'},  scatter_kws={'s': 20, 'alpha': 0.5})
plt.xlabel('Sentence Length')
plt.ylabel('Rouge Score')
plt.savefig("figs/t5-base_rouge_by_length.pdf", format="pdf")

# %%

test_predictions['data'] = test_predictions['modified_triple_sets'].map(lambda x: ";".join(x['mtriple_set'][0]))
worst_rouge = test_predictions.sort_values(by=['rouge']).tail().values
test_predictions.sort_values(by=['rouge']).tail()
# %%
print("\n\n".join([
    "\n".join([a for a in r if isinstance(a, str)])
    for r in 
    worst_rouge
]))

# %%
df['sentences'] = df["lex"].map(lambda x: " ".join(x['text']))
df['num_sentences'] = df["lex"].map(lambda x: len(x['text']))
df['sentences']
# %%
df['data'] = df['modified_triple_sets'].map(lambda x: ";".join(x['mtriple_set'][0]))
df['datalen'] = df['modified_triple_sets'].map(lambda x: x['mtriple_set'][0]).map(len)
df['data']
# %%
df['num_sentences'].describe()
# %%
df['sentences'].map(len).describe()
# %% we can plot the correlation between number of sentences and performance
ax = df[['num_sentences', 'datalen']].plot.box()
ax.set_xlabel("Corpus Attribute")
ax.set_ylabel("Count Per Record")
ax.set_xticklabels(["Sentences", "RDF Triples"])
plt.savefig("figs/box_plot_datacounts.pdf", format="pdf")
# %%
# %%
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")
# %%
df['datacharlen'] = df.data.map(len)
df['sentencecharlen'] = df.sentences.str.split(" ").map(len)

# %%
df['datatokenlen'] = df.data.map(tokenizer).map(lambda x: x['input_ids']).map(len)
df['sentencetokenlen'] = df.sentences.map(tokenizer).map(lambda x: x['input_ids']).map(len)
# %%
df['datatokenlen_p'] = df.data.str.replace("_"," ").map(tokenizer).map(lambda x: x['input_ids']).map(len)
df['sentencetokenlen_p'] = df.sentences.str.replace("_"," ").map(tokenizer).map(lambda x: x['input_ids']).map(len)
# %%
subattrs = df[[
    "datacharlen", "sentencecharlen",
    'datatokenlen', 'sentencetokenlen',
    'datatokenlen_p','sentencetokenlen_p'
]]
ax = subattrs.plot.box(figsize=(10,5), rot=15)
ax.set_xlabel("Corpus Attribute")
ax.set_ylabel("Length")
ax.set_xticklabels([
    "RDF Characters", "Sentence Words", 
     "RDF Tokens", "Sentence Tokens",
     "Normalized RDF Tokens","Normalized Sentence Tokens",
])
plt.savefig("figs/box_plot_tokencounts.pdf", format="pdf")
df.datatokenlen.describe()
# %%
import seaborn as sns

fig, ax = plt.subplots(figsize=(15, 5))

sns.violinplot(data=subattrs, ax=ax, fill=False, cut=0, width=.3, inner_kws=dict(box_width=5, color=".2"))
ax.set_xticklabels([
    "RDF Chars", "Sentence Words", 
     "RDF Tokens", "Sentence Tokens",
     "Normalized RDF Tokens","Normalized Sentence Tokens",
])
ax.set_xlabel("Corpus Attribute")
ax.set_ylabel("Length")
ax.set_yscale('log')
ax.set_xticklabels(ax.get_xticklabels(), rotation=15)  # Adjust the rotation angle as needed
plt.subplots_adjust(bottom=0.25)
plt.savefig("figs/violin_plot_tokencounts.pdf", format="pdf")

# %%

from datasets import load_dataset

dataset = load_dataset("wiki_bio")
# %%
dataset 
# %%
wb = pd.concat([
    pd.concat([pd.DataFrame(dataset[e]), pd.DataFrame([e] * len(dataset[e]), columns=['subset'])], axis=1) for e in ['val','train','test']], axis=0)
# %%
wb
# %%
wb['char count'] = wb['target_text'].map(len)
# %%
wb['word count'] = wb['target_text'].str.count(' ').map(lambda x: x + 1)

# %%
val_counts = pd.concat([
    df.subset.value_counts(),
    wb.subset.str.replace('val','dev').value_counts(),
], axis=1)
val_counts.columns = ['WebNLG', "WikiBio"]
val_counts = val_counts.append(val_counts.sum(axis=0), ignore_index=True)
print(val_counts.to_latex())
# %%
# %%
wb[['char count', 'word count']].describe()
# %%
wb
# %%
# %%
df
# %%
print(df[['sentences', 'data']].head(10).to_markdown())
# %%
print(df[['data']].tail().to_markdown())
# %%
relations = dict()
def append_triples(inpt: str):
    a = inpt.split(";")
    for b in a:
        b = b.split("|")
        if len(b) == 3:
            relation = b[1].strip()
            relations[relation] = relations.get(relation, 0) + 1

df['data'].map(append_triples)
# %%
# %%
",".join(relations)
# %%

rs = pd.Series(relations)
ax = rs.sort_values().plot()
ax.set_yscale('log')
ax.set_xticks([])
ax.set_ylabel("Frequency")
# %%
print(rs.sort_values(ascending=False).head(100).to_latex())
# %%

",".join( rs.sort_values().tail(100).keys())
# %%
import matplotlib.pyplot as plt
import numpy as np



# Function x**(1/2)
def forward(x):
    return x**(1/2)


def inverse(x):
    return x**2


plt.hist((rs.values), bins=100, density=False)
plt.xlabel('Label Occurrences')
plt.ylabel('Frequency')
# plt.xscale('function', functions=(np.exp, np.log))
# plt.yscale('function', functions=(np.exp, np.log))
plt.yscale('log')
ax.set_yscale('function', functions=(forward, inverse))

ax.set_xlim([1, 180])
# ax.set_ylim([1, 180])
plt.savefig("figs/log_relation_frequency.pdf", format="pdf")
# %%
from scipy.stats import kurtosis

# %%
kurtosis_value = kurtosis(rs.values)
kurtosis_value
# %%
np.log(10)
# %%
