from transformers import MarianMTModel, MarianTokenizer
from nltk.tokenize import sent_tokenize
import re

def clean_text(text):
    text = re.sub(r"\[[A-Z]+\]", "", text)
    text = re.sub(r"\((?:[0-9]+\/){2}[0-9]+\)", " ", text)
    if not re.search(r"\.", text):
        text = text.replace(",", ".")
#     text = re.sub(r"[^0-9a-zA-Z]+", " ", text)
#     text = re.sub(r"[^\w\s]", "", text)
    text = text.replace("'", "")
    text = text.replace('"', "")
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\n\n", " ", text)
    text = text.strip(" ")
    text = re.sub(' +',' ', text).strip() # gets rid of multiple spaces and replace with a single
    return text

def translate(text, model, tokenizer):
    if text is None or text == "":
        return "Error",

    #batch input + sentence tokenization
    batch = tokenizer.prepare_seq2seq_batch(sent_tokenize(text)).to('cuda')

    #run model
    translated = model.generate(**batch)
    tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    return " ".join(tgt_text)