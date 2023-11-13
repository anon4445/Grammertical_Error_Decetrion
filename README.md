# Grammertical_Error_Decetrion

This is a simple application that predicts the grammatical correctness of a given sentence using a pre-trained BERT model fine-tuned for sequence classification. It's implemented using Python and the Transformers library from Hugging Face.

## Getting Started

## Code Overview 

The project consists of two main components:

1. **Model Loading**
   - The BERT model (`BERT_GED`) and tokenizer are loaded using the Hugging Face Transformers library.
   
2. **Grammatical Correctness Prediction**
   - The user provides a sentence to be analyzed for grammatical correctness.
   - The model tokenizes the sentence and encodes it as input for the BERT model.
   - The model returns the predicted class label, indicating whether the sentence is grammatically "perfect" or "not right!!"

**Example**

For example, if you enter the sentence: "i am a student," the application will predict that it's not grammatically correct.



### Prerequisites

To run this project, make sure you have Python and the required libraries installed. You can install them using pip:

```bash
pip install transformers

