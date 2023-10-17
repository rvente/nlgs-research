# %%
from huggingface_hub import notebook_login

notebook_login()

# %%
import transformers

print(transformers.__version__)

model_checkpoint = "t5-small"
# %%
from datasets import load_dataset
from evaluate import load

raw_datasets = load_dataset("web_nlg", "release_v2")
metric = load("rouge")

raw_datasets
# %%
def fill_summary_and_document(training_sample):
    sp = dict(training_sample)
    document = "\n ".join(training_sample["lex"]['text'])
    summary = "\n ".join(training_sample['modified_triple_sets']['mtriple_set'][0])
    sp["document"] = document
    sp["summary"] = summary
    return sp

# %%
doc_sum_datasets = raw_datasets.map(fill_summary_and_document)

# %%
import datasets
import random
import pandas as pd
from IPython.display import display, HTML

# %%
metric
# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

tokenizer("Hello, this one sentence!")
# %%
tokenizer(["Hello, this one sentence!", "This is another sentence."])
print(tokenizer(text_target=["Hello, this one sentence!", "This is another sentence."]))
# %%
if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "summarize: "
else:
    prefix = ""
# %%
max_input_length = 1024
max_target_length = 128

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


preprocess_function(doc_sum_datasets['train'][:2])


# %%
tokenized_datasets = doc_sum_datasets.map(preprocess_function, batched=True)

# %%
import torch

# %%
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model = model.to(device)

# %%

# %%
batch_size = 16
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-webnlg",
    eval_steps=2000,
    evaluation_strategy = "steps",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
)

# %%
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# %%
import nltk
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Note that other metrics may not have a `use_aggregator` parameter
    # and thus will return a list, computing a metric for each sentence.
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
    # Extract a few results
    result = {key: value * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)


    return {k: round(v, 4) for k, v in result.items()}

# %%
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# %%
trainer.predict(tokenized_datasets['test'])

# %%
def text_to_prediction_single(text):
    return tokenizer.decode(trainer.predict([tokenizer(text)]).predictions[0])

t = "The FitnessGram Pacer Test is a multistage aerobic capacity test that progressively gets more difficult as it continues. The 20 meter pacer test will begin in 30 seconds. Line up at the start. The running speed starts slowly, but gets faster each minute after you hear this signal."
text_to_prediction_single(t)
# %%
# %%
t = "The leader of Aarhus is Jacob Bundsgaard."
tokenizer.decode(trainer.predict([tokenizer(t)]).predictions[0])
# %%

text_to_prediction_single("Torvalds was born in Helsinki, Finland,"
                          "the son of journalists Anna and Nils Torvalds")
# %%
