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
from huggingface_hub import notebook_login
from IPython.display import HTML, display
from transformers import (AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
                          GenerationConfig, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
assert torch.cuda.is_available()

# notebook_login()
NUM_TRAIN_EPOCHS = 10
TASK = 'd2s'
model_checkpoint = "t5-small"
NATURAL_LANGUAGE = "nl"
STRUCTURED_DATA = "sd"

TARGET = STRUCTURED_DATA if TASK == 'd2s' else NATURAL_LANGUAGE
INPUT = NATURAL_LANGUAGE if TASK == 'd2s' else STRUCTURED_DATA
assert TARGET != INPUT
# %%
# %%
df = pd.read_pickle("~/repos/nlgs-research/pipeline/normalized_data/webnlg_clean.pkl")
df
# %%

metric = combine([
    load("rouge"),
    load("bleu"),
    load('meteor'),
])
# %%
df
# %%
metric
# %%

# %%
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# %%
max_input_length = 256
max_target_length = 256

def preprocess_function(examples):
    inputs = examples["document"]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["label"] = labels["input_ids"]
    return model_inputs

tokenize = lambda x: tokenizer(x, max_length = max_input_length, truncation=True, padding=True)
tokenized = tokenize(list(df[INPUT].values))
# %%
# %%
df['input_ids'] = tokenized['input_ids']
df['attention_mask'] = tokenized['attention_mask']
df['label'] = df[TARGET].map(tokenize).map(lambda x: x['input_ids'])

# TODO: figure out what the target length should be and if it's actually training on the right data
# https://huggingface.co/docs/transformers/v4.29.1/en/tasks/translation#translation

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
batch_size = 16
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"models/{model_name}-finetuned-webnlg-{TASK}-2e-4",
    eval_steps=500,
    generation_config=generation_config,
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
    generation_max_length=2048,
    generation_num_beams=4,
)

# %%
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# %%
def compute_metrics(eval_pred):
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

    # Note that other metrics may not have a `use_aggregator` parameter
    # and thus will return a list, computing a metric for each sentence.
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)#, use_stemmer=True, use_aggregator=True)

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    ret = result
    print(ret)
    p = Path("snapshots/metrics")
    t = p.read_text()
    p.write_text(t + "\n" + str(ret))
    
    return ret

# %%
df['input_ids'].map(len)
# %%
def pd_to_dataset(df: pd.DataFrame, split='train') -> Dataset:
    d = Dataset.from_pandas(df[df.subset== split ][['input_ids','attention_mask','label']], split=split)
    return d.remove_columns("__index_level_0__")
pd_to_dataset(df, 'train')
# %%
model(np.array([1, 2]))
# %%
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=pd_to_dataset(df, 'train')['input_ids'],
    eval_dataset=pd_to_dataset(df, 'test')['label'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# %%
try:
    trainer.train(resume_from_checkpoint=True)
except ValueError as e:
    trainer.train()

# %%
predictions = trainer.predict(tokenized_datasets['dev'])
print(predictions)
# %%
predictions = trainer.predict(tokenized_datasets['test'])
predictions
# %%
def text_to_prediction_single(text):
    return tokenizer.decode(trainer.predict([tokenizer(text)]).predictions[0])

# %%
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
test_predictions = pd.DataFrame(tokenized_datasets['test'])
prediction_ids = pd.Series(list(predictions.predictions))
test_predictions['predicted'] = (prediction_ids
        .map(list)
        .map(lambda x: [a for a in x if a != -100 and a != 0])
        .map(tokenizer.decode)
)
rouge = load('rouge')
# %%
test_predictions['rouge'] = (
    (test_predictions['predicted'].map(lambda x: [x]) + test_predictions['summary'].map(lambda x: [x]))
    .map(lambda x: rouge.compute(references=[x[0]], predictions=[x[1]], use_stemmer=False, use_aggregator=False, rouge_types=['rouge2']))
    .map(lambda x: x['rouge2'][0])
)
                
# %%
# test_predictions['rouge'] = rouge.compute(predictions=test_predictions['predicted'], references=test_predictions['summary'])
# %%