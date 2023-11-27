# %%
from huggingface_hub import notebook_login
import torch
assert torch.cuda.is_available()

# notebook_login()
NUM_TRAIN_EPOCHS = 10
TASK = 'd2s'
# %%
import transformers

print(transformers.__version__)

model_checkpoint = "t5-small"
# %%
from datasets import load_dataset
from evaluate import load, combine

raw_datasets = load_dataset("web_nlg", "release_v2")
metric = combine([
    load("rouge"),
    load("bleu"),
    load('meteor'),
])
raw_datasets
# %%
fst_samp = raw_datasets["train"][0]
fst_samp

# %%
def fill_summary_and_document(training_sample):
    sp = dict(training_sample)
    sentences = " ".join(training_sample["lex"]['text'])
    data = " ".join(training_sample['modified_triple_sets']['mtriple_set'][0])

    # Data to sentence
    # sp["document"] = data
    # sp["summary"] = sentences

    # sentence to data
    sp["document"] = sentences
    sp["summary"] = data

    del sp['category']
    del sp['lex']
    del sp['original_triple_sets']
    del sp['modified_triple_sets']
    return sp
# %%
fill_summary_and_document(fst_samp)
# %%
doc_sum_datasets = raw_datasets.map(fill_summary_and_document)

# %%
import datasets
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=5):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df[["modified_triple_sets", 'lex']].to_html()))

# %%
show_random_elements(raw_datasets["train"])
# %%
metric
# %%
# fake_preds = ["hello there", "general kenobi"]
# fake_labels = ["hello there", "general kenobi"]
# metric.compute(predictions=fake_preds, references=fake_labels)

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# %%
if model_checkpoint in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
    prefix = "summarize: "
else:
    prefix = ""
# %%
max_input_length = 256
max_target_length = 256

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["label"] = labels["input_ids"]
    return model_inputs


preprocess_function(doc_sum_datasets['train'][:2])

# TODO: figure out what the target length should be and if it's actually training on the right data
# https://huggingface.co/docs/transformers/v4.29.1/en/tasks/translation#translation

# %%
tokenized_datasets = doc_sum_datasets.map(preprocess_function, batched=True)

# %%
import torch

# %%
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, GenerationConfig
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model = model.to(device)

# %%
generation_config = GenerationConfig.from_pretrained(model_checkpoint)
generation_config.min_length = 5
generation_config.max_length = 2048
generation_config.early_stopping = True
# generation_config.pad_token_id
generation_config.no_repeat_ngram_size = 5
generation_config.temperature = .9



# %%
batch_size = 16
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-webnlg-{TASK}-1e-4",
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
import nltk
import numpy as np
import os
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "true"

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

    # TODO: checkme
    # bertscore = load("bertscore")
    # bscore = {
    #     ("bertscore"+k):
    #     (np.mean(v) if not isinstance(v, str) else v)
    #     for k,v in bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang='en').items()
    # }

    # ret = {**result, **bscore}
    ret = result
    print(ret)
    p = Path("snapshots/metrics")
    t = p.read_text()
    p.write_text(t + "\n" + str(ret))
    
    return ret

# %%
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# %%
trainer.train(resume_from_checkpoint=True)
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
# %% so it must be a count in characters, not tokens. Good to know.

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
test_predictions.to_pickle("metrics/" + model_checkpoint +'-'+ TASK + '-test.pkl')
# %%

Path(f"metrics/{model_checkpoint}-{TASK}_log_hist").write_text(str(trainer.state.log_history))
# %%
# %%
df = pd.DataFrame(raw_datasets['train'])
df['text'] = df['modified_triple_sets'].map(lambda x : " ".join(x['mtriple_set'][0]))
# %%
df['tokenlen'] = df['text'].map(tokenizer).map(lambda x: x['input_ids']).map(len)
# %%
df['tokenlenprime'] = df['text'].str.replace("_", " ").map(tokenizer).map(lambda x: x['input_ids']).map(len)
# %%
