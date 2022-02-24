#  CAPSTONE: SENTIMENT ANALYSIS
***BY: KEVIN LUU***

[WEB APP DEMO](https://share.streamlit.io/l-kevin/sentiment_analysis/main/webapp/main_app.py)

## EXECUTIVE SUMMARY

### PROBLEM STATEMENT
Dread calling into customer service to fix your issue....
What are these big corps doing?

Are there common trends?
Can it be generalized?


*Citations:*
<br><sup>1</sup> - https://www.theglobeandmail.com/life/article-why-the-pandemic-videogame-boom-is-great-for-canada/
<br><sup>2</sup> - https://www.washingtonpost.com/video-games/2020/05/12/video-game-industry-coronavirus/
<br><sup>3</sup> - https://www.theverge.com/2020/5/13/21257227/coronavirus-streamelements-arsenalgg-twitch-youtube-livestream-numbers


### DATA
- Sourced from [Stanford University](http://ai.stanford.edu/~amaas/data/sentiment/) -- collected of IMDB reviews
- Training dataset consists of 25000 labeled reviews
  - 12500 negative reviews (<= 4 ratings)
  - 12500 positive reviews ()>=7 ratings )
- Unsupervised (unlabeled) dataset of 50000 reviews
  - inclusive of between 4-7 ratings
- Originally in the form of 75000 .txt files:
  - compiled into MS Excel Queries --> transformed & exported as UTF-8 .csv file

### METHODOLOGY
- Data Wrangling
    - data manipulation to comma-separated-values
    - review text cleaning
      - (lowercase, removal of URL elements, retained letters only)
- Experimental Data Analysis (EDA)
    - NLP count vectorization for most frequent words per sentiment
      - horizontal bar chart
      - word cloud
    - character and word count distributions
    - word2vec - top 30 similar words to bad/good
- Modeling Training & Implementation
    - 3 models
      - Logistic Regression (LogReg)
        - TF-IDF Vectorizer
      - Convolutional Neural Network (CNN)
      - Bidirectional Autoregressive Tranformer (BART)
        - Pre-trained models:
          - (local) [FaceBook's bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli)
          - (web-app server) [Valhalla's distilbart-mnli-12-3](https://huggingface.co/valhalla/distilbart-mnli-12-3) for server deployment

**Data Cleaning:**
<br>The feature 'selftext' was the only one that had missing values. This was due to the post being an image-only post without any text. Removing these posts was not the option as the title of the posts still contained useful information. Therefore, all null posts along with all posts that only had "[deleted]" or "[removed]" were filled with a blank space.
<br>Since the title also contained useful information, a new column called 'text' was created as a combination and replacement of the 'title' and 'selftext' features
<br>The 'subreddit' feature contained 'pcgaming' or 'console' -- therefore, was handled with label-encoding:
- pcgaming = 1
- console = 0
<br>The cleaned data was then exported into a new .csv file as it no longer has any null values, the text information has been consolidated, and the target feature is now numerical. The data was ready for natural language processing & modeling.

**Data Dictionary:**
|Feature|Dataset|Description|
|:---|:---|:---|
|subreddit|submissions_clean.csv|The subreddit that the post belongs to (0 = r/consoles, 1 = r/pcgaming)|
|text|submissions_clean.csv|The consolidated text of the title and contents of the post.|

**EDA:**
<br>With the use of Count Vectorization on the consolidated text, the initial look at the top 20 features indicated that there were many commonly shared words between both subreddits & that there were URL elements that needed to be omitted.
<br>This resulted in additional text cleaning to remove all URL elements, remove non-letter characters, and lower-case everything.
<br>Upon a second look at the top 20 words after the text cleaning, additional commonly shared words were discovered. Therefore, a custom stop words list that included the commonly shared words was created.
<br>Here's a visual look at the top 20 words from each subreddit:
![top20console](images/top20_console.png "Top 20 words in r/consoles")
![top20pc](images/top20_pc.png "Top 20 words in r/pcgaming")

**Modeling:**
<br>In order to select the best model for training, the 5-fold cross validation scores all 14 combinations of 7 models + 2 vectorizers (Count and TF-IDF) were compared:
- Logistic Regression
- KNN
- BernoulliNB
- MultinomialNB
- Random Forest Classifier
- SVC (Support Vector Classifier)
- AdaBoost Classifer

The top models considered for hyperparameter tuning were:
1. Logistic Regression
2. Random Forest Classifier
3. SVC
4. AdaBoost Classifier

**Table of scores of tuned models:**
|  | Count Vectorizer | TD-IDF Vectorizer |
|---|---:|---:|
| **Logistic Regression** | Training: 0.98154<br>Testing: 0.90713<br>ROC_AUC: 0.88226 | Training: 0.90812<br>Testing:0.89746<br>ROC_AUC: 0.87036 |
| **Random Forest Classifier** | Training: 0.99661<br>Testing: 0.89443<br>ROC_AUC: 0.87812 | Training: 0.99633<br>Testing: 0.90169<br>ROC_AUC: 0.88252 |
| **Support Vector Classifier** | Training: 0.97728<br>Testing: 0.90371<br>ROC_AUC: 0.87972 | Training: 0.98373<br>Testing: 0.91156<br>ROC_AUC: 0.89170 |
| **Ada Boost Classifier** | Training: 0.90534<br>Testing: 0.89705<br>ROC_AUC: 0.86792 | Training: 0.90703<br>Testing: 0.89142<br>ROC_AUC: 0.86241 |

*NOTE: This table is the simplified version without the best parameters. Please refer to [eda_and_modeling.ipynb](https://git.generalassemb.ly/kluu/project-3/blob/master/code/eda_and_modeling.ipynb) for the table with tuned parameters.*

**Through tuning, the best 2 models are:**
1. **AdaBoost Classifier with Count Vectorization (lowest variance)**
- Training: 0.90534
- Testing: 0.89705
- ROC_AUC: 0.86792


- Parameters:
    - stop_words = 'english'
    - ngram_range = (1,2)
    - DecisionTreeClassifier(max_depth=1)
    - n_estimators = 200
    - learning_rate = 0.5


2. **SVC with TF-IDF Vectorization (highest ROC_AUC score + lowest bias)**
- Training: 0.98373
- Testing: 0.91156
- ROC_AUC: 0.89170


- Parameters:
    - stop_words = stops (custom, extending English list)
    - ngram_range = (1,2)
    - kernel = sigmoid
    - gamma = 0.1
    - C = 10


Some additional tuning was performed and it turns out that there was another SVC model that performed slightly better:
<br>**Support Vector Classifier with TF-IDF Vectorizer**
- Training: 0.98373
- Testing: 0.91156
- ROC_AUC: 0.89170


- Parameters:
    - kernal='linear'
    - C=1
    - gamma=1

**Feature importance:**
<br>While looking at the feature importances of each of the top 2 models, depending on the stop word treatment, different words and combinations of words had an impact on the models performance:
1. AdaBoost + Count
- many combinations of games + "word" had an impact on the models performance in predicting the PC Gaming subreddit.
    - this was to be expected the model was tuned to only remove the original 'English' stop words instead of the custom one


2. SVC + TF-IDF
- more distinctive single words have a heavier level of importance (along with some two word combinations that consist of the listed single words)
    - since the custom stop words list was used for removal, we no longer see combinations of "games" + "words" compared to the previous list


### CONCLUSION
To achieve the goal of correctly predicting whether a post belongs to subreddit r/pcgaming or r/consoles, one of the two models can be utilized:
1. **ADABOOST CLASSIFIER with COUNT VECTORIZATION**
    - lowest variance for consistent predictions and best generalization


2. **SUPPORT VECTOR CLASSIFER with TF-IDF VECTORIZATION**
    - highest roc_auc for strong separability performance (and lowest bias)

### RECOMENDATIONS
1. Include lemmatization/stemming in the tuning process
    - Potentially better scores may exist and the feature importances could narrow down
2. Implement image processing due to many posts being image-only
3. Re-conduct the analysis when the r/consoles community has expanded to have a minimum of 10000 posts
    - Due to imblanced samples from each subreddit (more r/pcgaming vs. r/consoles)
