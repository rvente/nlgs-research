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
df = pd.read_pickle("~/repos/nlgs-research/pipeline/normalized_data/webnlg_clean.pkl")
df
# %%
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# %%
max_input_length = 256
max_target_length = 256
tokenize = lambda x: tokenizer(x, max_length = max_input_length, truncation=True, padding=True)
tokenized = tokenize(list(df[INPUT].values))
# %%
# [markdown]
# The following fields comprise the "interface" of the model, despite the fact
# the documentation doesn't make this obvious. 
# %%

df['input_ids'] = tokenized['input_ids']
df['attention_mask'] = tokenized['attention_mask']
df['labels'] = df[TARGET].map(lambda x: [tokenize(e)['input_ids'] for e in x])

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
batch_size = 32 if model_checkpoint == "t5-small" else 16
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"models/{model_name}-finetuned-webnlg-{TASK}-2e-4",
    eval_steps=1000,
    evaluation_strategy = "steps",
    learning_rate=2e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=6,
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
df['input_ids'].map(len)
# %%
def pd_to_dataset(df: pd.DataFrame, split='train') -> Dataset:
    if split != 'train':
        d = df[df.subset== split ][['input_ids','attention_mask','labels']].head(30)
        d.labels =d.labels.map(get[0])
        d = Dataset.from_pandas(d, split=split)
        return d.remove_columns("__index_level_0__")
    else:
        distribute_x_over_y = lambda x, y: seq([x]).cartesian(y).to_list()
        d = df[df.subset== split ][['input_ids','attention_mask','labels']]
        # there's a simplified way to do this but this is was more test-able.
        # TODO: write me!
        if TASK == 'mt':
            # interlaced data - be sure to prepend task specifier
            df 

        if TASK == 'd2s':
            pairings = (
            seq(d.input_ids)
                .zip(d.attention_mask)
                .zip(d.labels)
                .starmap(distribute_x_over_y)
                .map(lambda x: [x[0][0][0], x[0][0][1], x[0][1] ])
            )
            pair_df = pd.DataFrame(pairings, columns=['input_ids','attention_mask','labels'])
            return Dataset.from_pandas(pair_df)
        if TASK == 's2d':
            

        
pd_to_dataset(df, 'train')
# %%
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=pd_to_dataset(df, 'train'),
    eval_dataset=pd_to_dataset(df, 'dev'),
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
predictions = trainer.predict(pd_to_dataset(df,'test'))
predictions
# %%
def text_to_prediction_single(text):
    return tokenizer.decode(trainer.predict([tokenizer(text)]).predictions[0])

# %%
pred_df = pd.DataFrame(columns=['pred_ids'], data=pd.Series(list(predictions.predictions)))
pred_df['decoded'] = pred_df.pred_ids.map(lambda x: [e for e in x if e > 1]).map(tokenizer.decode)
pred_df['subset'] = 'test'

pred_df = pred_df.reset_index()
pred_df
# %%
pred_df.to_pickle(f"~/repos/nlgs-research/pipeline/predictions/{TASK}-{model_name}-{NUM_TRAIN_EPOCHS}.pkl")
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