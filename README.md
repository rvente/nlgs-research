# Towards a bijection between prose and structured data

Graduate coursework, with the paper "Inroads to natural language structured data bijection and the role of LLM annotated data" in the `paper/` folder.

See the "continue the project" section below or the "future work" in the paper for future research directions. 

A variant of the models trained is published here https://huggingface.co/vente/t5-small-finetuned-webnlg-mt-2.0e-04


One may prompt it with 

`s2d 0:  Torvalds was born in Helsinki, Finland, the son of journalists Anna and Nils Torvalds`

or

`d2s 0: Torvalds|birth place|Finland;` 

and should obtain sensible output generations. But note that the default generation settings of huggingface may be different from those used in the paper.

## Structure

1. Data downloading is taken care of by the hugging face datasets library
  - `preproc.py` should run first to clean the data (preprocessing)
2. Training baselines and experiments is controlled by the global variables
   at the top of `finetune.py`
   - reads the data emitted from preprocessing.py and trains on them
3. By changing the global vars and running `finetune.py` again
4. Scoring the model has the `analysis_*.py` and is partitioned by sub-task

## Structure

<details>
  <summary>Code Structure</summary>

```
├── README.md
├── analysis_d2s.py # data to sentence evaluation
├── analysis_mt.py  # multi-task evaluation
├── analysis_s2d.py # sentence to data evaluation
├── anaysis_corpus.py # compute corpus statistics
├── finetune.py # trains the networks, saving results in models/ outputting predictions to pipelines/predictions
├── preproc.py
├── cuda-envs
│   ├── base_requirements.txt
│   ├── [...]
│   └── env.yaml
├── models
│   ├── t5-base-finetuned-webnlg-d2s-2.0e-04
│   ├── [...]
│   └── t5-small-finetuned-webnlg-s2d-2.0e-04
├── pipeline
│   ├── anaysis_corpus.py    # pre-midterm analysis including plots
│   ├── normalized_data      # store and reuse raw and pre-processed versions of the corpora
│   ├── figs # stores the figures emitted by the analysis_corpus.py
│   │   ├── box_plot_datacounts.pdf
│   │   ├── [...]
│   │   └── violin_plot_tokencounts.pdf
│   ├── predictions  # save all the predictions themselves in pickle files
│   │   ├── d2s-t5-base-5.pkl
│       ├── [...]
│   │   └── s2d-t5-small-5.pkl
│   └── scores # plots, and score csv's are output here by model
│       ├── d2s-t5-base-5  # some logs are also provided
│       ├── s2d-t5-base-5
│       └── s2d-t5-small-5
└── funcutils.py # a bespoke small library I wrote for convenience functions
```

</details>


### History 

This code repos makes extensive use of Jupyter Code Cells within the Python Interactive window

https://code.visualstudio.com/docs/python/jupyter-support-py

The file `finetune.py` started as the official huggingface summarization example. Then was incrementally
re-written until it worked for the WebNLG task.

https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization.ipynb

### Installing

A short but incomplete list of things to install. One may choose to use the cuda environments for installing instead.

```
pip install datasets evaluate transformers rouge-score nltk sentencepiece

huggingface-cli login # in case you want to use any "push to hub" commands

pip install -r base_requirements.txt # run from cuda-envs directory

python3 -c "import torch; print(torch.cuda.is_available()); print(torch.__version__)" # should print "True" if cuda is installed correctly.
```


### Useful Commands

One might accidentally commit one of the larger pickle files. For example `mt-t5-base-5.pkl` was committed because it generated output that was too long and over the 10MB limit. 

https://stackoverflow.com/questions/8083282/how-do-i-remove-a-big-file-wrongly-committed-in-git

```
conda env list
conda create --name ngls --clone torch
conda activate ngls
conda env export > env.yaml
```

Sources For the useful commands above

- https://stackoverflow.com/a/48947404
- https://stackoverflow.com/questions/41274007/anaconda-export-environment-file
- https://stackoverflow.com/a/59456960

## Continue the project

This uses t5 because it's a good experimental platform because it doesn't take too long to train.

Low hanging fruit for semantic parsing task
- What if the model I trained is just too "detailed" or not "detailed" enough? Extend the scoring logic to extract true positive rate and false positive rate to see if it's being penalized for extracting too many relations (even if they are correct) 
- How does the model perform when trained on LLM annotated wikibio data and tested on webnlg data? This would be a true test of generalization because the corpora are qualitatively different.
- Similar to above but using templates to create fake training data. Can we extract 100 or so templates from the existing samples? How does the model fare with parsing in this case
