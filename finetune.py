# %%
import os
import random
from pathlib import Path

import datasets
import nltk
import numpy as np
import pandas as pd
import torch
import transformers

from datasets import Dataset, concatenate_datasets
from evaluate import combine, load
from functional import seq
from funcutils import get
from huggingface_hub import notebook_login
from IPython.display import HTML, display
from transformers import (AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
                          GenerationConfig, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
assert torch.cuda.is_available()

# notebook_login()
NUM_TRAIN_EPOCHS = 5
TASK = 's2d' # or 's2d' or 'mt' pull from argv
model_checkpoint = "t5-small"

NATURAL_LANGUAGE = "nl"
STRUCTURED_DATA = "sd"

TARGET = NATURAL_LANGUAGE if TASK == 'd2s' else STRUCTURED_DATA 
INPUT = STRUCTURED_DATA if TASK == 'd2s' else NATURAL_LANGUAGE 
assert TARGET != INPUT
# %%
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# %%
max_input_length = 256
max_target_length = 256
tokenize = lambda x: tokenizer(x, max_length = max_input_length, truncation=True, padding=True)
# %%
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model = model.to(device)

# %%
generation_config = GenerationConfig.from_pretrained(model_checkpoint)
generation_config.min_length = 5
generation_config.max_length = 2048
generation_config.early_stopping = True
generation_config.no_repeat_ngram_size = 5
generation_config.temperature = .9

# %%
batch_size = 64 if model_checkpoint == "t5-small" else 16
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"models/{model_name}-finetuned-webnlg-{TASK}-2e-4",
    eval_steps=1000,
    evaluation_strategy = "steps",
    learning_rate=2e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size//2,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
    save_steps=600,
    generation_config=generation_config,
    generation_max_length=2048,
    generation_num_beams=4,
    # generation_no_repeat
)

# %%
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# %%
metric = combine([
    load("rouge"),
    # load("bleu"),
    # load('meteor'),
])
metric
# %%
def compute_metrics(eval_pred, task):
    # Note since this is a sequence-to
    predictions, labels = eval_pred
    print(predictions)
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    #444 Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    ret = result
    p = Path("snapshots/metrics")
    t = p.read_text()
    p.write_text(t + "\n" + str(ret))
    
    return ret

# %%
df = pd.read_pickle("~/repos/nlgs-research/pipeline/normalized_data/webnlg_clean.pkl")
df
# %%
# we must invent `seed_number` since d2s can output multiple sentences for the
# same data input. So the seed will be a generation parameter, in the case that
# we are working in a deterministic environment, so generation can vary as
# desired.

cartesian_sd_nl = []
for (i, subset, cat, indx, sd, nl) in df.itertuples():
    for j, candidate in enumerate(nl):
        pairing = dict(
            record_idx=i,
            seed_number=j,
            subset=subset,
            category=cat,
            split_index=indx,
            sd=sd,
            nl=candidate,
        )
        cartesian_sd_nl.append(pairing)
# calling this "flattened" because it no longer has nested records
flt = pd.DataFrame(cartesian_sd_nl)

# %%
# prepend the seed number. This should be rt of the prompt hereafter for `d2s`
# tasks. So, prompting with two different numbers should never generate the same
# output.
if TASK == "d2s":
    flt['sd'] = flt.seed_number.map(lambda x: "<" + str(x) + "> ")  + flt.sd
flt
# %%
tokenized = tokenize(list(flt[INPUT].values))
# %%  [markdown]
# !!WARNING!! The following fields comprise the "interface" of the model,
# despite the fact the documentation doesn't make this obvious. Without these
# particular names, ['input_ids', 'attention_mask', 'labels'],
# the model will not train and provide cryptic error messages
# %%
flt['input_ids'] = tokenized['input_ids']
flt['attention_mask'] = tokenized['attention_mask']
flt['labels'] = flt[TARGET].map(lambda x: tokenize(x)['input_ids'])
flt['input_ids'].map(len)
# %%
flt.input_ids.sample(300).map(str).to_json('vis.json')
# %%
def pd_to_dataset(df: pd.DataFrame, split='train') -> Dataset:
    d = df[df.subset== split ][['input_ids','attention_mask','labels']]
    return Dataset.from_pandas(d)
        
# get_ds alias should bake in the desired argument. Makes you wish python
# supported currying
get_ds = lambda x: pd_to_dataset(flt, x)
get_ds('train')
# %%
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=get_ds('train'),
    eval_dataset=get_ds('dev'),
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=lambda x: compute_metrics(x, 'd2s'),
)

# %%
# we try-catch because resume_from_checkpoint returns a value error (?!)
# if training did not begin first.
try:
    trainer.train(resume_from_checkpoint=True)
except ValueError as e:
    print(e)
    trainer.train()

# trainer.train()
# %%
# %%
predictions = trainer.predict(get_ds('test'))
predictions
predictions
# %%
def text_to_prediction_single(text):
    return tokenizer.decode(trainer.predict([tokenizer(text)]).predictions[0])

# %%
flat_keep_positive = lambda x: [e for e in x if e > 1]
pred_df = pd.DataFrame(columns=['pred_ids'], data=pd.Series(list(predictions.predictions)))
test_set = flt[flt.subset == 'test']
decoded = pred_df.pred_ids.map(flat_keep_positive).map(tokenizer.decode)
test_set['decoded'] = decoded
pred_df['decoded'] = decoded
test_set['pred_ids'] = pred_df.pred_ids
pred_df['subset'] = 'test'

pred_df = pred_df.reset_index()
pred_df
# %%
save_fname = f"~/repos/nlgs-research/pipeline/predictions/{TASK}-{model_name}-{NUM_TRAIN_EPOCHS}.pkl"
test_set.to_pickle(save_fname)
save_fname
# %% [markdown]
# ## Sanity Checks
# %%
t = "The leader of Aarhus is Jacob Bundsgaard."
tokenizer.decode(trainer.predict([tokenizer(t)]).predictions[0])
# %%

text_to_prediction_single("Linus Torvalds was born in Helsinki, Finland,"
                          "the son of journalists Anna and Nils Torvalds")
# %%
print("\n".join(map(text_to_prediction_single, [
    "<pad> United_States | leaderName | Barack_Obama </s>",
    "<pad> 'Anderson,_Indiana | isPartOf | Fall_Creek_Township,_Madison_County,_Indiana', 'Fall_Creek_Township,_Madison_County,_Indiana | country | United_States', 'Anderson,_Indiana | isPartOf | Indiana' </s>"
])))
# %%

print("\n".join(map(tokenizer.decode,
                np.where(predictions.predictions != -100, predictions.predictions, tokenizer.pad_token_id)
                )))
# %%
# %%
max(map(len, predictions.predictions))
# %%
predictions.predictions
# %%