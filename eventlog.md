https://huggingface.co/docs/transformers/model_doc/t5

https://colab.research.google.com/github/patil-suraj/exploring-T5/blob/master/t5_fine_tuning.ipynb


https://github.com/liu-hz18/Prompt-GLM/blob/main/train_t5.sh

https://github.com/ericakcc/PEFT-T5-for-Dialogue-Summarization/blob/main/Fine_Tune_T5_for_Dialogue_Summarization.ipynb
'

The followint looks promising
https://github.com/siddhantmedar/Fine-tuned-FLAN-T5-based-Model-for-Summarization-Task/blob/main/PEFT%20fine-tuned%20FLAN-T5-XXL.ipynb


The current task is: get something working on colab first


https://chat.openai.com/share/4eaafe65-ce5f-4b8e-bac2-8ece26f120af


FINALLY FOUND ONE

https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization.ipynb#scrollTo=uNx5pyRlIrJh


    train_dataset=tokenized_datasets["train"].select(range(2000)),
    eval_dataset=tokenized_datasets["validation"].select(range(500)),

https://stackoverflow.com/a/48947404

 conda env list
# conda environments:
#
base                  *  /home/vente/.conda
tf                       /home/vente/.conda/envs/tf
torch                    /home/vente/.conda/envs/torch
    
conda create --name ngls --clone torch
conda activate ngls

https://stackoverflow.com/questions/41274007/anaconda-export-environment-file

https://stackoverflow.com/a/59456960

 conda env export > env.yaml


 pip install datasets evaluate transformers rouge-score nltk sentencepiece



 huggingface-cli login


 As of f452504 on 9 Oct, first test finetuning is underway but on a different summarization task, using the xsum dataset.

 Next step is to inspect the existing webnlg dataset on hugging face and then figuring out how to create my own.

|   Step   | Training Loss | Validation Loss | Rouge1   | Rouge2   | Rougel  | Rougelsum | Gen Len   |
|:--------:|:-------------:|:---------------:|:--------:|:--------:|:-------:|:---------:|:---------:|
|  2000    |   2.814100    |     2.560796    | 26.772300 | 6.755400 | 20.918600 | 20.911200  | 18.814300 |
|  4000    |   2.781000    |     2.522533    | 27.531000 | 7.227500 | 21.592400 | 21.588700  | 18.840200 |


as of a42f888 on 14 October, summarization is underway for the text to data subtask

|   Step   | Training Loss | Validation Loss | Rouge1   | Rouge2   | Rougel  | Rougelsum | Gen Len   |
|:--------:|:-------------:|:---------------:|:--------:|:--------:|:-------:|:---------:|:---------:|
|  2000    |   0.787000    |     0.486647    | 52.231400 | 38.483700 | 48.844900 | 48.861900  | 18.172300 |
|  4000    |   0.543000    |     0.335245    | 54.225600 | 43.192300 | 51.581900 | 51.527400  | 18.126600 |