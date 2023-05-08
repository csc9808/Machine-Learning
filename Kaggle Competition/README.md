# Introduction
Twitter has been widely used as a medium to communicate disasters. Throughout this project, we aimed to train a natural-language classifier that classifies tweets as disasters or non-disasters. For achieving this purpose, we utilized the pre-trained models from [Hugging Face](https://huggingface.co/) and fine-tuned them to better suit our task. 


We decided to use pre-trained transformers models not only because of their convenience, but also because of the variety of models it provided. As we will see soon, the pre-trained model that we used was already trained on twitter texts, meaning that it already had knowledge about how tweets differ from, for example, a journal report. This task is also part of a [Kaggle competition](https://www.kaggle.com/competitions/nlp-getting-started), in which we are also participating. 


However, we noticed that participants were using different base models for the competition. Some people used other transformers models, such as BERT and RoBERTa, while others made a PyTorch/TensorFlow model from scratch. Especially those who used pre-trained models used models that were trained on natural language sentences, which differ from tweets. Such heavy preprocessing scripts may have been necessary for models that were not trained on tweets, but given that our model was trained tweet data, we decided to reduce the amount of data preprocessing significantly. 



### Model: Limits and Future Directions

We fine-tuned the pre-trained model using Hugging Face's Trainer API, and the API did not provide means of reporting the accuracy as it was training. In addition, the test dataset provided by the Kaggle competition did not contain target values, so we had no way of evaluating the model's performance on new data other than the evaluation loss reported above. Therefore, we had to evaluate the accuracy after the model was trained, on the same training data used for fine-tuning the model. While we believe this itself as a major limitation of our project, it makes sense that the test dataset did not have any labels, for the datasets were provided to serve the purposes of a competition. 

Meanwhile, because the dataset was small we assumed that a larger dataset with more text examples could perhaps help lower the losses and increase the accuracies. Additionally, given the similar losses and accuracies between the raw model and processed model, we began to wonder if there were any other, more significant factors that we would have missed. Considering these limitations, we believe some future directions for this project can include the following: 

1. Trying different data pre-processing methods
2. Implementing a training loop from scratch to report the accuracy as the model trains
3. Increasing the size of the training dataset so the model can learn from more data



### Model: Conclusions

We created two versions of the training set: (1) raw text and (2) text after pre-processing. We applied both to the model we trained and stored the results in the new column of each dataframe. When the results were compared with the given target value, the first set of raw text had an accuracy of 84.63%, while the second set of preprocessed data had a slightly lower accuracy of 84.07%. As indicated by the results of the testing the models, using raw text without removing url and replacing mentions returned a slightly higher accuracy by 0.56%. This computes to about 40 tweets labeled more accurately out of 7613 that were tested. 

In addition, we trained two additional Naive-Bayes classifiers on the raw text data and the processed text data. While we expected the NB models to have a drastically lower accuracy measure than the transformers models, the baseline classifiers had accuracy values of 78.79% and 78.86%, which we thought was a decent value for a probability distribution based model not pre-trained on tweets. 

Although the differences are minute, the results were different from what we anticipated as the texts were pre-processed with the expectation of getting a more accurate results than using the raw text.


# Conclusion

The purpose of this project was to build a model that determines whether or not the given tweet indicates a real disaster. The training data set contained id, keyword, location, text, and target of each tweet. We further expanded the data set by extracting hashtags, mentions, and links from the text and appending to new columns. The newly included information was used for data visualization and comparison between disaster and non-disaster tweets. Then the model was fine-tuned with both raw and processed texts from the training set, which resulted in 84.63% and 84.07% accuracy respectively. Then the comparison between raw and processed text was evaluated using the test data, which came out to be 99.79% equivalence between the two types of texts. In terms of general uses, we believe this model can be utilized by news organizations, insurance companies, and government agencies to respond to disasters quickly and accurately.
