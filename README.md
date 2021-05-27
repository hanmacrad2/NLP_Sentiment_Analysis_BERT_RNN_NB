### Sentiment Analysis of Amazon Reviews - Comparison of Classification methods

AI Group project for module CS7IS2 of TCD

#### Abstract 
In this work, three distinct methods of classification were implemented to predict the polarity of Amazon reviews. 

Across several domains, a wide range of models have been used to detect sentiment and our aim was to test a sample of this range and compare the efficacy of the more traditional models with the state-of-the-art. Accordingly the following models were implemented and compared;

- BERT model
- LSTM-RNN model
- Naïve-Bayes model

BERT is a state-of-the-art pre-trained model that is beating every NLP benchmark and is the architecture being used to revamp both Google Search and Microsoft Bing engines. Here we tested the hypothesis that the BERT model is significantly better than earlier DNNs such as LSTMs or more traditional classifiers such as Naïve Bayes. We did this by performing sentiment analysis on labelled Amazon review data and comparing the models' performance across metrics such as the ROC curve and accuracy.

#### Methods

The problem involved determining the polarity of Amazon reviews, specifically whether they are positive or negative, using three distinct models, BERT, Naïve Bayes and RNN to ascertain which method had the highest efficacy.

The models were trained on one dataset and tested on another. Originally, the training dataset consisted of 1,800,000 reviews and the test dataset 200,000. However, they were reduced to 100,000 training reviews and 20,000 test reviews to reduce training time. The datasets were balanced, containing equal amounts of positive and negative reviews. Standardised preprocessing was implemented to ensure consistency across the model results. This included the removal of stopwords and non-alphabetical characters. There are two labels, positive (2) and negative (1) which were updated to 1 and 0 respectively to be compatible with the sklearn package. This was carried out for each of the three models; BERT, LSTM-RNN and Naive Bayes in the respective packages above, i.e;

- BERT_Transformer_model
- rnn
- naive_bayes

#### Conclusion
The BERT model was found to have the best predictive performance, however at the cost of a prohibitively slow training time. Possible improvements to the training time could be made by optimising the computing approach. Graphics Processing Units (GPUs) can significantly accelerate the training process for deep learning models by taking advantage of a GPU's massively parallel architecture. Thus for future implementation of this study, the program could be designed to offload tasks to one or more GPUs potentially reducing the training time from days to hours. Despite it's simplicity, the Naïve-Bayes model had by far the superior training time, taking merely 2 minutes to train.  

Overall, if the speed of model execution is favoured over small gains in accuracy, Naive Bayes should be the chosen model over it's competitors. However if model accuracy and performance is the priority, the BERT model should be the default model of choice.

