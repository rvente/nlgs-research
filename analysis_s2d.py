# %%
# this file should take in all files and produce the appropriate s2d analysis
from pathlib import Path


list(Path("./pipeline/predictions").glob("*s2d*"))

# %%
rouge = load('rouge')
test_predictions['rouge'] = (
    (test_predictions['predicted'].map(lambda x: [x]) + test_predictions['summary'].map(lambda x: [x]))
    .map(lambda x: rouge.compute(references=[x[0]], predictions=[x[1]], use_stemmer=False, use_aggregator=False, rouge_types=['rouge2']))
    .map(lambda x: x['rouge2'][0])
)
                
# %%
# test_predictions['rouge'] = rouge.compute(predictions=test_predictions['predicted'], references=test_predictions['summary'])
# %%