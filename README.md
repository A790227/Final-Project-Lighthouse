# LLM Project
## Sentiment Analysis of Amazon top profitable category (Beauty & Personal Care: 39% of sellers) 

## Project Task
- Sentiment Analysis of Amazon top profitable category (Beauty & Personal Care Reviews):
    The goal of the project is to classify reviews from McAuley-Lab Amazon-Reviews-2023 dataset into positive or negative sentiments.
    The task involves using a pre-trained language model to fine-tune it for binary classification.

## Dataset
- Dataset Used: McAuley-Lab Amazon-Reviews-2023 dataset:
The dataset contains:
This is a large-scale Amazon Reviews dataset, collected in 2023 by McAuley Lab, and it includes rich features such as:

- User Reviews dataset Beauty_and_Personal_Care(ratings, text, helpfulness votes, etc.); 10.2 GB
Item Metadata dataset Beauty_and_Personal_Care(descriptions, price, raw image, etc.); 2.63GB
Links (user-item / bought together graphs).

- Source: Hugging Face datasets library (McAuley-Lab Amazon-Reviews-2023 dataset)

## Sample sizes 

- First pre-model was run with Formula to Calculate Sample Size:
For a given confidence level, margin of error, and proportion:

        Where:
        n: Required sample size (for infinite population)
        Z: Z-score corresponding to the desired confidence level

        For example:
        90% confidence level → 𝑍 = 1.645
        95% confidence level → Z=1.96
        99% confidence level → Z=2.576

        p: Estimated proportion of the population with the characteristic of interest (default 𝑝=0.5 if unknown, as it maximizes sample size)
        E: Margin of error (e.g., E=0.05 for ±5%)
- sample sizes 385 registers

- Second pre-model was run with optimal sizes calculate Increase Sampling Proportion
Rather than sampling the statistically minimal sample size, increase the proportion of the dataset being sampled. For example, if your calculated sample size is 385, sampling 0.4% of the dataset 
- sample sizes 95.857 registers

## EDA - Exploratory Data Analysis 


- User Reviews dataset Beauty_and_Personal_Care(ratings, text, helpfulness votes, etc.); 10.2 GB (EDA REVIEW DATASET)
- Item Metadata dataset Beauty_and_Personal_Care(descriptions, price, raw image, etc.); 2.63GB (EDA META DATASET)


- Process: Examine column names, data types, memory usage, and size. Understand the structure and basic attributes of the dataset. Distributions

- Process: Plot histograms of rating, helpful_vote, and review_length.
    -- Goal: Understand the spread and distribution of key numerical features.
    Correlations

- Process: Compute and visualize correlations between rating, helpful_vote, and review_length.
    Goal: Identify relationships between numerical features.
    Temporal Trends

- Process: Convert timestamp to datetime and analyze trends in rating over time.
    Goal: Explore changes in ratings over the years.
    Verified Purchases Analysis

- Process: Compare rating and helpful_vote between verified and non-verified purchases.
    Goal: Understand how verified status affects ratings and helpfulness.
    Product Ratings

- Process: Calculate and visualize average ratings for products (asin), highlighting top and bottom performers.
    Goal: Identify the best and worst-rated products.
    Relationships

- Process: Explore relationships between review_length, rating, and helpful_vote.
    Goal: Determine how review length impacts ratings and helpfulness.
    Word Analysis

- Process: Perform word frequency analysis and generate a WordCloud.
    Goal: Identify common themes and keywords in review texts.


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
- First pre-model stadistical formula to Calculate Sample Size (Minimal)

    - Epoch 3 ran successfully, and it seems you completed training with:
    - Training Loss: 0.594529
    - Validation Loss: 0.501252
    - Accuracy: 0.792208 (or ~79.22% validation accuracy)

These results indicate that the model performs well on classifying reviews with balanced performance between positive and negative sentiments. However. THE PREDICTION RESULTS LACK CONFIDENCE DUE TO THE SMALL SAMPLE SIZE, WHICH PROVIDES INSUFFICIENT STATISTICAL POWER.

- Second pre-model optimal sizes calculate Increase Sampling Proportion (Optimal)

    - Epoch 3 ran successfully, and it seems you completed training with:
    - Training Loss: 0.2024 (low, indicating a good fit to the training data).
    - Validation Loss: 0.1846 (low, suggesting the model generalizes well).
    - Validation Accuracy: 0.9339 (~93.39%, excellent performance).

These results indicate that the model performs well on classifying reviews with balanced performance between positive and negative sentiments. 
THE PREDICTION RESULTS is CONFIDENCE.
  

## Results 
- The model was trained for 3 epoch , and it achieved an accuracy of 93.39% on the validation set, which is good result for a basics pass through the dataset.
- The training loss (0.2024) is slightly more than the validation loss (0.1846), which is expected because the model is usually more accurate on the data it was trained on. The small gap indicates that the model is not significantly overfitting.
- Overall, the model is performing well with high accuracy and reasonable loss values, suggesting it's making good predictions on both the training and validation sets after 3 epoch with optimal sizes calculate Increase Sampling Proportion .

## Links 
- https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- https://www.junglescout.com/resources/articles/amazon-product-categories/


