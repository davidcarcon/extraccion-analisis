from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

task='emotion'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)

def saca_score(text):
	encoded_input = tokenizer(text, return_tensors='tf')
	output = model(encoded_input)
	scores = output[0][0].numpy()
	scores = softmax(scores)
	return scores

