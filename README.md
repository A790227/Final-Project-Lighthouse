# LLM Project
## Sentiment Analysis of IMDb Movie Reviews

## Project Task
- Sentiment Analysis of IMDb Movie Reviews:
    The goal of the project is to classify movie reviews from the IMDb dataset into positive or negative sentiments. The task involves using a pre-trained language model to fine-tune it for binary classification.

## Dataset
- Dataset Used: IMDb Movie Reviews:
The dataset contains 50,000 movie reviews labeled as either positive or negative. The dataset is already split into a training set and a test set, with 25,000 reviews in each set.

- Source: Hugging Face datasets library (imdb dataset)
- Training Set: 25,000 labeled reviews
- Test Set: 25,000 labeled reviews

## Pre-trained Model
- Model Used: DistilBERT
    The pre-trained model selected for this project is DistilBERT, a smaller and faster version of BERT that retains over 97% of BERT’s language understanding capabilities while being more efficient.

    Model Name: distilbert-base-uncased-finetuned-sst-2-english
    Model Source: Hugging Face Model Hub
    Task: Fine-tuned for sentiment analysis, binary classification (positive/negative)
## Performance Metrics
- Metrics Used:

    - Accuracy: Measures the proportion of correctly classified reviews out of the total reviews.
    - F1 Score: A balance between precision and recall, important for imbalanced datasets.
    - Precision and Recall: Used to understand how well the model is at identifying positive and negative reviews.

Results:
    - Accuracy on Test Set: e.g., 89.20%
    - F1 Score (Positive): e.g., 0.90
    - F1 Score (Negative): e.g., 0.89
These results indicate that the model performs well on classifying movie reviews with balanced performance between positive and negative sentiments.
## Hyperparameters
- Most Relevant Hyperparameters:
    - Learning Rate: A learning rate of 2e-5 was found to work well, balancing model updates without overshooting.
    - Batch Size: A batch size of 16 was chosen to balance between training speed and computational resource availability.
    - Number of Epochs: The model was fine-tuned for 1 epoch. This was sufficient to achieve high performance, as the dataset was relatively small and the model was already pre-trained.
    

## Results 
- The model was trained for just 1 epoch, and it achieved an accuracy of 92.68% on the validation set, which is good result for a single pass through the dataset.
- The training loss (0.1851) is slightly lower than the validation loss (0.2352), which is expected because the model is usually more accurate on the data it was trained on. The small gap indicates that the model is not significantly overfitting.
- Overall, the model is performing well with high accuracy and reasonable loss values, suggesting it's making good predictions on both the training and validation sets after just 1 epoch.

## Links 
- https://huggingface.co/A790227
- https://colab.research.google.com/github/A790227/data-project-llm/blob/main/untitled0.ipynb


