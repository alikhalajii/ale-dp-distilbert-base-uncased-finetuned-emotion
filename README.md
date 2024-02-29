# Emotion Classification with DistilBERT

This repository contains a fine-tuned variant of `distilbert-base-uncased` for emotion classification. The model was trained on the emotion dataset and has demonstrated strong performance in evaluations.

## Model Details

- **Model Name**: ale-dp/distilbert-base-uncased-finetuned-emotion
- **Performance**: [Check the [Hugging Face Model Page](https://huggingface.co/ale-dp/distilbert-base-uncased-finetuned-emotion) for detailed performance metrics and evaluation results.]


## Usage

### Loading the Model

To use the pre-trained model in your Python script or Jupyter Notebook, you can use the following code:

```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

model_name = "ale-dp/distilbert-base-uncased-finetuned-emotion"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)


from transformers import pipeline
import numpy as np

# Load the model from Hugging Face
model = "ale-dp/distilbert-base-uncased-finetuned-emotion"
classifier = pipeline("text-classification", model=model)

# Define class labels
class_labels = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]

def emotion_classifier(text):
    # Get predictions
    preds = classifier(input_tweet, return_all_scores=True)

    # Extract scores
    scores = preds[0]['scores']

    # Find the index with the highest score using argmax
    max_score_index = np.argmax(scores)

    # Return the predicted label
    return class_labels[max_score_index]

my_text = "I'm in such a happy mood today i feel almost delighted and i havent done anything different today then i normally have it is wonderful"

predicted_label = emotion_classifier(my_text)
print(f"The predicted emotion for the text is: {predicted_label}")

