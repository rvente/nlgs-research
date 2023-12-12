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
|  6000    |   0.463900    |     0.285747    | 55.065900 | 45.050700 | 52.667100 | 52.642300  | 18.132800 |
|  8000    |   0.437800    |     0.273149    | 55.363300 | 45.575900 | 52.929100 | 52.901400  | 18.139000 |


still need a way to emit predictions so that bertscore may be computed
still need to validate that the output is correct

I did that. It works!


now I need to figure out the parameters to pass into the model. Learning rate and decay. 


Training with 2e-5 for 100 epochs gave the following results in about 4.5 hours

'rouge1': 0.426801639515604, 'rouge2': 0.3368091317409858, 'rougeL': 0.3959538960945993, 'rougeLsum': 0.4093016460133704, 'bleu': 0.029955359605418112, 'precisions': [0.9017084632614519, 0.7314420803782505, 0.591418770160258, 0.480850826261724], 'brevity_penalty': 0.045518364365731145, 'length_ratio': 0.2445203346327738, 'translation_length': 22769, 'reference_length': 93117, 'gen_len': 18.927733168622606, 'bertscoreprecision': 0.9191489429515111, 'bertscorerecall': 0.8748397045182622, 'bertscoref1': 0.8961523412700934, 'bertscorehashcode': 'roberta-large_L17_no-idf_version=0.3.12(hug_trans=4.24.0)'}

I changed to 2e-4 and doubled the batch size


pip install ipywidgets


I had to refresh my conda env and build it from scratch https://anaconda.org/anaconda/cudatoolkit
conda install -c anaconda cudatoolkit


I should have made a back up of the environment before deleting it...
good thing I have a backup of my base env

pip install -r base_requirements.txt

python3 -c "import torch; print(torch.cuda.is_available()); print(torch.__version__)"

We're back in business.

...
right now it's saying that the tokenizer doens't like the output
of the decode...

out of range integral type conversion attempted
https://github.com/huggingface/transformers/issues/22634

so I filtered out the -100 tokens

from the prediction


Found a bug! I have to take one and only one of the sentences?
Or evaluate BLEU score properly with all alternatives.



42,567 Characters	$0.02
 

Curiosity: error analysis - is the model robust to label occurences or does it generalize?

Checklist

- normalize the data
- create evaluation code for palm... wait I should already have some code I could re-use for that use-case...
- Extract the wikibio segments and get them labeled
    - de


For my memory fragmentation issue:

https://huggingface.co/docs/datasets/v2.15.0/en/package_reference/main_classes#datasets.Dataset.from_generator.keep_in_memory


the data is normalized. 

now I want to define the different subsets from the data as reversed elements for the hybrid model and save that. Later I will need to subset to non-hybrid use-cases


Now finetine.py is great. I will find a subset that the model performs particularly badly on, and generate synthetic data with that subset in mind.


I actually _can_ use their models and have an apples-to-apples comparison
As long as I do the same pre-procesing.


Great. I finished training the small versions and I fixed a bug.

5 December

I need to write analysis d2s.py which will simply re-use evaluation code I have already written.



Multitasking up next

now I need to train the multi-task version, and be sure to save
the predictions from each of the subsets




1. Find the most common relations for artists and politicians
1. At first _any_ prompt for synthetic data generation will suffice
1. d2t and t2d networks on synthetic data (naive)


principled prompt engineering
1. Figure out why it's so slow
1. evaluate on webnlg random subsample of 50 (at first)


I discovered I could use junix to extract images from jupyter notebooks
but I wish I could save them all as pdf. Perhaps I could deal with doing so via vs code
or parsing the html post-hoc but that is out of scope.

pip install junix


also parameterizing notebooks is possible - so I could in principle have a single
notebook to run everything.


Before submitting, I would like to do a re-run of everything
just but I guess this is not supposed to come before my paper submission.


Refactoring incrementally in between complete runs?


Need to perform a join to complete the 


eval code for d2s

train the medium variant, produce data and plots

train a single large variant from the best performer thus far

Quantify word overlap, generate synthetic data, train the best model with this current dara, filter out generations which are sufficiently different. Add in synthetic data, and check if it all works out. At least report back numbers for semantic parsing of webnlg, even if you don't use this for prompt refinement.


Where am i? 

just training the last model before palm's various generation steps. Use bertscore similarity between subjects to determine what wikibio data to input for the weakest categories. Also generate from n-gram similarities 'astronaut' or weakest class. Just train one final model on this.


https://stackoverflow.com/questions/8083282/how-do-i-remove-a-big-file-wrongly-committed-in-git