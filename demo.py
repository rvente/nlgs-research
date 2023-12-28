# %%
import random
import gradio as gr

from transformers import T5Tokenizer, T5ForConditionalGeneration, GenerationConfig
import torch

MDL = "vente/t5-small-finetuned-webnlg-mt-2.0e-04"
tokenizer = T5Tokenizer.from_pretrained(MDL)
model = T5ForConditionalGeneration.from_pretrained(MDL)
generation_config = GenerationConfig.from_pretrained(MDL)

# the following 2 hyperparameters are task-specific
def alternatingly_agree(message, history):
    max_source_length = 512
    max_target_length = 512

    # Suppose we have the following 2 training examples:
    input_sequence_1 = message
    input_sequences = [input_sequence_1]

    encoding = tokenizer(
        input_sequences,
        padding="longest",
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

    # forward pass
    output = model.generate(input_ids,
        early_stopping=True,
        num_beams=5,
        max_new_tokens=1024,
        temperature=.9
    )
                            
    decoded = tokenizer.decode([x for x in output[0] if x>0])
    print(decoded)
    return decoded

gr.ChatInterface(alternatingly_agree).launch()
# %%
