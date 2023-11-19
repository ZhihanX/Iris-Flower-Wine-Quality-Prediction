# ID2223 Lab1 Iris & Wine-quality

This is the course content of KTH ID2223 HT23 Scalable Machine Learning and Deep Learning. For more information please visit the course webpage: https://id2223kth.github.io/

# Task1: Iris Flower Classification
## 1.1 Requirements for Task1
In Task1, our tasks are:

1) Build and run a feature pipeline on Modal

2) Run a training pipeline

3) Build and run an inference pipeline with a Gradio UI on Hugging Face Spaces.

Pipeline picture:
![image](https://github.com/ZhihanX/Wine-quality/assets/114545801/1d65de66-052a-42ee-a078-51a04ece7844)

## 1.2 Steps and Explanation

First of all, we create a new environment via HOPSWORKS, Modal and Hugging face. Both the latter use the API_KEY from Hopsworks. Then we run the Serverless Machine Learning pipeline with Iris Flower Dataset. As for the Iris Flower Dataset, we want to predict the variety, given the length and width of the petal and sepal. And classify Iris Flowers with the K-Nearest Neighbor algorithm. We have four files for the pipeline, we run "iris-eda-and-backfill-feature-group.ipynb" first to get some basic information about Iris.csv and do some preprocessing works and then we upload it to Hopsworks. We run "iris-feature-pipipeline-daily.py" secondly which will generate a new synthetic flower randomly based on our dataset via Modal to train the model. And we run the "iris-training-pipeline.ipynb" thirdly to download the features and labels, get the model with KNN and upload it. Finally, we run "iris-batch-inference-pipeline.py" to do prediction and upload our classification pictures, prediction table and confusion matrix. 

After all these, we complete a good pipeline and we can run "Iris UI app.py" which includes the Gradion UI on Hugging Face Space and we can input and feature number we want to predict. Also, we can download the logs we update and show them all together on Hugging face with "Iris Monitor UI app.py"



# Task2: Wine Quality Classification
## 2.1 Requirements for Task2
In Task2, our tasks are:

1) Write a feature pipeline notebook that registers the wine quality dataset as a Feature Group with Hopsworks.

2) Write a training pipeline that reads training data with a Feature View from Hopsworks, trains a regression or classifier model to predict if a wine’s quality. Register the model with Hopsworks.

3) Write a Gradio application that downloads your model from Hopsworks and provides a User Interface to allow users to enter or select feature values to predict the quality of a wine for the features you entered.

4) Write a synthetic wine generator function and write a new “daily” feature pipeline that runs once per day to add a new synthetic wine.

5) Write a batch inference pipeline to predict the quality of the new wine(s) added, and build a Gradio application to show the most recent wine quality prediction and outcome, and a confusion matrix with historical prediction performance. 

## 2.2 Dataset
Wine Quality (https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/wine.csv)

## 2.3 Steps and Explanation
Pipeline picture:
![image](https://github.com/ZhihanX/Wine-quality/assets/114545801/02dfba17-33fd-4c85-a17d-3544593d01d7)

### 2.3.1 Preprocessing & Adjust the classifying label ([wine_with_hopsworks_backfill.ipynb](https://github.com/ZhihanX/Wine-quality/blob/main/Task2%20wine/wine_with_hopsworks_backfill.ipynb))
We initially utilized a series of functions to observe and analyze the source dataset. After deduplication and removal of NaN values, we observed that the labels 3, 4, 5, 6, 7, 8, 9 were overly concentrated around the center of the labels, which are classes 5, 6, and 7. Consequently, the machine learning model built on this classification label yielded only around 55% accuracy. Therefore, we decided to adjust the classification labels by grouping 3, 4, 5, 6 together as 'bad' and 7, 8, 9 together as 'good'. We also encoded the column 'type' which encodes white as 0, and red as 1. After all the preprocessing and adjustments, the data frame is uploaded to the Hopsworks.

### 2.3.2 Add daily synthetic wine ([wine-feature-pipeline-daily.py](https://github.com/ZhihanX/Wine-quality/blob/main/Task2%20wine/wine-feature-pipeline-daily.py))
We defined a range for bad quality and good quality wine. The “daily” feature pipeline runs once per day to add a new synthetic wine randomly chosen from the range we defined.

### 2.3.3 Train classifier model with Random Forest ([wine_training_pipeline.ipynb](https://github.com/ZhihanX/Wine-quality/blob/main/Task2%20wine/wine_training_pipeline.ipynb))
We compared the training accuracy locally with the following code:
```python
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(random_state=rdm_value),
    'SVM': SVC(),
    "SVM (linear)": SVC(kernel="linear"),
    "SVM (polynomial)": SVC(kernel="poly"),
    "Gradient Boosting": GradientBoostingClassifier(random_state=rdm_value),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'Logistic Regression': LogisticRegression(),
    'BaggingClassifier': BaggingClassifier(random_state=rdm_value),
    'MLP': MLPClassifier(),
    'Naive Bayes': GaussianNB()
}
```
We got the best performance with Random Forest. The predicting accuracy could reach over 80% when the proportion of the dataset to include in the test split is 0.2.

![image](https://github.com/ZhihanX/Wine-quality/assets/114545801/8f801c83-ea13-4fa4-87d7-2778efad80a2)


### 2.3.4 Predict new added wine & get historical prediction information ([wine-batch-interence-pipeline.py](https://github.com/ZhihanX/Wine-quality/blob/main/Task2%20wine/wine-batch-interence-pipeline.py))
Predict the quality of the new added wine with the trained model in 2.3.3. We also found 2 image URLs for this part and connected them with the prediction results. Meanwhile, a confusion matrix with historical prediction performance is contained in this part.

### 2.3.5 User Interface
Here are our UI URLs on Hugging Face:

Hugging-face:https://huggingface.co/spaces/PatrickML/Wine

Hugging-face-monitor:https://huggingface.co/spaces/PatrickML/wine_monitor

# Collaborators
[Zhihan Xu](https://github.com/ZhihanX)

[Conglei Xiang](https://github.com/CLxxxxxxxxara)
