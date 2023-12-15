# %%
import os
import gc
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
from huggingface_hub import notebook_login
from IPython.display import HTML, display
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, GenerationConfig,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

from funcutils import get

os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
assert torch.cuda.is_available()

NUM_TRAIN_EPOCHS = 5
IS_MULTI_CORPUS = True
TASK = 'mt' # 'd2s' or 's2d' or 'mt' pull from argv
MODEL_CKPNT = "t5-base" # t5-small or t5-base
NATURAL_LANGUAGE = "nl"
STRUCTURED_DATA = "sd"
MULTI_CORP = '-multicorp' if IS_MULTI_CORPUS else ""
LR = 2.0e-4
TRAIN_CHKPNT_NAME = f"models/{MODEL_CKPNT}-finetuned-webnlg-{TASK}-{LR:.1e}{MULTI_CORP}"

TARGET = NATURAL_LANGUAGE if TASK == 'd2s' else STRUCTURED_DATA 
INPUT = STRUCTURED_DATA if TASK == 'd2s' else NATURAL_LANGUAGE 
TRAIN_CHKPNT_NAME
# %% unbind the variables
assert TARGET != INPUT
del NATURAL_LANGUAGE
del STRUCTURED_DATA
# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPNT)
# %%
max_input_length = 256
max_target_length = 256
tokenize = lambda x: tokenizer(x, max_length = max_input_length, truncation=True, padding=True)
tokenize
# %%
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CKPNT)
model = model.to(device)
model
# %%
generation_config = GenerationConfig.from_pretrained(MODEL_CKPNT)
generation_config.min_length = 5
generation_config.num_beams = 4
generation_config.max_length = 2048
generation_config.early_stopping = True
generation_config.no_repeat_ngram_size = 2
generation_config.temperature = .9

# %%
batch_size = 64 if MODEL_CKPNT == "t5-small" else 16
# START: ADAPTED FROM https://huggingface.co/docs/transformers/tasks/summarization
args = Seq2SeqTrainingArguments(
    TRAIN_CHKPNT_NAME,
    eval_steps=1500,
    evaluation_strategy = "steps",
    learning_rate=LR,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size//2,
    gradient_accumulation_steps=2 if MODEL_CKPNT != 't5-small' else 1, # so we have an effective batch size of 32
    weight_decay=0.01,
    save_total_limit=5,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
    save_steps=600,
    generation_config=generation_config,
    generation_max_length=200,
)
# END: ADAPTED FROM https://huggingface.co/docs/transformers/tasks/summarization
# %%
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
metric = combine([
    load("rouge"),
])
metric
# %%
# START: COPIED FROM https://huggingface.co/docs/transformers/tasks/summarization
def compute_metrics(eval_pred):
    # monitor memory and force gc. probably slows us down, probably 
    torchmem = torch.cuda.memory_allocated()
    torchcap = torch.cuda.get_device_properties(0).total_memory

    print(f"torch has allocated {torchmem} of {torchcap}")

    predictions, labels = eval_pred
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return result
# END: COPIED FROM https://huggingface.co/docs/transformers/tasks/summarization
# %%
df = pd.read_pickle("~/repos/nlgs-research/pipeline/normalized_data/webnlg_clean.pkl")

if IS_MULTI_CORPUS:
    df = pd.read_pickle("~/repos/nlgs-research/pipeline/normalized_data/webnlg_wikibio_joint.pkl")
df
# %% [markdown]
# we must invent `seed_number` since d2s can output multiple sentences for the
# same data input. So the seed will be a generation parameter, in the case that
# we are working in a deterministic environment, so generation can vary as
# desired. This computes a cartesian product.

# %%
cartesian_sd_nl = []
for (i, subset, cat, indx, sd, nl) in df.itertuples():
    for j, nl_option in enumerate(nl):
        pairing = dict(
            record_idx=i,
            seed_number=j,
            subset=subset,
            category=cat,
            split_index=indx,
            sd=sd,
            nl=nl_option,
            task=TASK if TASK != 'mt' else 's2d' 
        )
        cartesian_sd_nl.append(pairing)
        if TASK == "mt":
            reverse_pair = pairing.copy()
            reverse_pair['sd'] = nl_option
            reverse_pair['nl'] = sd
            reverse_pair['task'] = 'd2s'
            cartesian_sd_nl.append(reverse_pair)

# calling this "flattened" because it no longer has nested records
has_not_run = True
flt = pd.DataFrame(cartesian_sd_nl)
flt
# %%
# prepend the seed number. This should be rt of the prompt hereafter for `d2s`
# tasks. So, prompting with two different numbers should never generate the same
# output.

if (TASK == "mt") and has_not_run:
    has_not_run = False
    flt['sd'] = flt.task + flt.seed_number.map(lambda x: " " + str(x) + ": ") + flt.sd

    # allow the model to code switch between corpora
    if IS_MULTI_CORPUS:
        flt['sd'] = flt.category.map(lambda x: 'wb' if x == 'WikiBio' else "") + flt.sd
flt
# %%
tokenized = tokenize(list(flt[INPUT].values))
# %%  [markdown]
# !!Heads-up!! The following fields comprise the "interface" of the model,
# despite the fact the documentation doesn't make this obvious. Without these
# particular names, ['input_ids', 'attention_mask', 'labels'],
# the model will not train and provide cryptic error messges. 
# %%
flt['input_ids'] = tokenized['input_ids']
flt['attention_mask'] = tokenized['attention_mask']
flt['labels'] = flt[TARGET].map(lambda x: tokenize(x)['input_ids'])
flt['input_ids'].map(len)
# %%
flt
# %%
# this will keep only the needed fields in memory on the GPU
def pd_to_dataset(df: pd.DataFrame, split='train') -> Dataset:
    print(df)
    d = df[df.subset== split][['input_ids','attention_mask','labels']]
    return Dataset.from_pandas(d)
        
# get_ds alias should bake in the desired argument. Makes you wish python
# supported currying
get_ds = lambda x: pd_to_dataset(flt, x)
tds = get_ds('train')
eds = get_ds('dev')
tds
# %%
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tds,
    eval_dataset=eds,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# %%
# we must try-catch because resume_from_checkpoint throws a value error (for
# some reason instead of raising a warning) if training did not begin first.
try:
    trainer.train(resume_from_checkpoint=True)
except ValueError as e:
    print(e)
    trainer.train()
# %%
if False:
    trainer.push_to_hub()
# %%
try:
    del tds
    del eds
    del tds
except Exception as e:
    print(e)
# %%
tds = get_ds('test')
# debug = Dataset.from_dict(tds[0:2])
predictions = trainer.predict(tds)
predictions
# %%
flat_keep_positive = lambda x: [e for e in x if e > 1]
pred_df = pd.DataFrame(columns=['pred_ids'], data=pd.Series(list(predictions.predictions)))
decoded = pred_df.pred_ids.map(flat_keep_positive).map(tokenizer.decode)
pred_df['decoded'] = decoded
pred_df['subset'] = 'test'
pred_df
# %%
test_set = flt[flt.subset == 'test'].copy()
test_set['pred_ids'] = list(pred_df['pred_ids'].values)
test_set['decoded'] = list(pred_df['decoded'].values)
test_set
# %%
save_fname = f"~/repos/nlgs-research/pipeline/predictions/{TASK}-{MODEL_CKPNT}-{NUM_TRAIN_EPOCHS}{MULTI_CORP}.pkl"
test_set.to_pickle(save_fname)
save_fname
# %% [markdown]
# ## Sanity Checks
# %%
def text_to_prediction_single(text):
    tensors = tokenizer("<pad>" + text + "</s>", return_tensors='pt').to(device)['input_ids']
    generation = trainer.model.generate(tensors,
        early_stopping=True,
        num_beams=5,
        max_new_tokens=1024,
        temperature=.9,
    ) 
    return tokenizer.decode(generation[0], skip_special_tokens=True)

t = "The leader of Aarhus is Jacob Bundsgaard."
text_to_prediction_single(t)
# %%
print("\n".join(map(tokenizer.decode,
                np.where(predictions.predictions != -100, predictions.predictions, tokenizer.pad_token_id)
                )))
# %%
if TASK == "mt":
    print("\n".join(map(text_to_prediction_single, [
        'd2s 0: Aarhus|leader name|Jacob Bundsgaard',
        'd2s 1: Aarhus|leader name|Jacob Bundsgaard',
        "d2s 0: United States|leader name|Barack Obama ",
        's2d 0: The leader of Aarhus is Jacob Bundsgaard.',
        "s2d 0: Linus Torvalds was born in Helsinki, Finland. He is the son of journalists Anna and Nils Torvalds",
        "s2d 1: Linus Torvalds was born in Helsinki, Finland. He is the son of journalists Anna and Nils Torvalds",
    ])))
else:
    print("\n".join(map(text_to_prediction_single, [
        'Aarhus|leader name|Jacob Bundsgaard',
        'Aarhus|leader name|Jacob Bundsgaard',
        "United States|leader name|Barack Obama ",
        'The leader of Aarhus is Jacob Bundsgaard.',
        "Linus Torvalds was born in Helsinki, Finland. He is the son of journalists Anna and Nils Torvalds",
        "Linus Torvalds was born in Helsinki, Finland. He is the son of journalists Anna and Nils Torvalds",
    ])))
    
# %%