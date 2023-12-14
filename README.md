# README

## Structure

<details>
  <summary>Code Structure</summary>

```
├── README.md
├── analysis_d2s.py # data to sentence evaluation
├── analysis_mt.py  # multi-task evaluation
├── analysis_s2d.py # sentence to data evaluation
├── anaysis_corpus.py 
├── finetune.py # trains the networks, saving results in models/ outputting 
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
│   │   ├── webnlg_clean.pkl
│   │   ├── webnlg_raw.pkl
│   │   └── wikibio.pkl
│   ├── figs # stores the figures emitted by the analysis_corpus.py
│   │   ├── box_plot_datacounts.pdf
│   │   ├── [...]
│   │   └── violin_plot_tokencounts.pdf
│   ├── predictions  # save all the predictions themselves in pickle files
│   │   ├── d2s-t5-base-5.pkl
│       ├── [...]
│   │   └── s2d-t5-small-5.pkl
│   └── scores # better named logs, is where intermediate files are saved
│       ├── d2s-t5-base-5
│       │   ├── analysis_d2s.ipynb
│       │   ├── d2s_scores.pkl
│       │   └── finetune.ipynb
│       ├── [...]
│       ├── s2d-t5-base-5
│       │   ├── analysis_s2d.ipynb
│       │   └── finetune.ipynb
│       └── s2d-t5-small-5
│           └── analysis_s2d.ipynb
└── funcutils.py
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

One might accidentally commit one of the larger pickle files. For example `mt-t5-base-5.pkl` was committed because
it generated output that was too long and over the 10MB limit. 

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