from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


def load_generation_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model.to(device)


def load_filtering_model(model_name, device):
    NLI_tokenizer = AutoTokenizer.from_pretrained(model_name)
    NLI_model = AutoModelForSequenceClassification.from_pretrained(model_name)

    return NLI_tokenizer, NLI_model.to(device)
