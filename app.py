import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def load_model():
    tokenizer = BertTokenizer.from_pretrained("BERT_GED")
    model = BertForSequenceClassification.from_pretrained("BERT_GED")
    return model, tokenizer

def predict(model, tokenizer, sentence):
    # Tokenize sentence
    encoded_dict = tokenizer.encode_plus(
        sentence, 
        add_special_tokens=True,
        max_length=64, 
        padding="max_length",
        truncation=True,
        return_attention_mask=True, 
        return_tensors='pt', 
    )
    input_ids = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']

    # Model inference
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    index = torch.argmax(logits, -1).item()  # Get the predicted class (0 or 1)

    if index == 1:
        return "perfect"
    else:
        return "not right!!"
    
def main():
    st.title("Grammatical Correctness Predictor")
    sentence = st.text_area("Sentence to analyze:")

    if st.button("Analyze"):
        if sentence:
            model, tokenizer = load_model()
            prediction = predict(model, tokenizer, sentence)
            st.write(f'"{sentence}" is grammatically {prediction}')
        else:
            st.warning("Please enter a sentence.")

if __name__ == "__main__":
    main()
