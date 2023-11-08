import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, BertConfig  
model_loaded = BertForSequenceClassification.from_pretrained("BERT_GED")
tokenizer = BertTokenizer.from_pretrained("BERT_GED")
sent = "i am a student"
print(sent)
encoded_dict = tokenizer.encode_plus(
        sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=64,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )
input_id = encoded_dict['input_ids']
print(input_id)
attention_mask = encoded_dict['attention_mask']
print(attention_mask)
input_id = torch.LongTensor(input_id)
attention_mask = torch.LongTensor(attention_mask)
with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model_loaded(input_id, attention_mask=attention_mask) 
logits = outputs[0]
print(logits)
index = logits.argmax()
print(index)
torch_label = index.detach().cpu().numpy()
print(torch_label)