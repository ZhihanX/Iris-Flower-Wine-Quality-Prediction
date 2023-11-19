# Iris & Wine-quality

Hugging-face:https://huggingface.co/spaces/PatrickML/Wine

Hugging-face-monitor:https://huggingface.co/spaces/PatrickML/wine_monitor

Task1: In Task1, our tasks are:

1) Build and run a feature pipeline on Modal

2) Run a training pipeline

3) Build and run an inference pipeline with a Gradio UI on Hugging Face Spaces.

Pipeline picture:
![image](https://github.com/ZhihanX/Wine-quality/assets/114545801/1d65de66-052a-42ee-a078-51a04ece7844)



First of all, we create a new environment via HOPSWORKS, Modal and Hugging face. Both the latter use the API_KEY from Hopsworks. Then we run Serverless Machine Learning pipeline with Iris Flower Dataset. As for Iris Flower Dataset, we want to predict the variety, given the length and width of the petal and sepal. And classify Iris Flowers with K-Nearest Neighbor algorithm. We have four files for the pipeline, we run "iris-eda-and-backfill-feature-group.ipynb" firstly to get some basic information of Iris.csv and do some precessing works and then we upload it to Hopsworks. We run "iris-feature-pipipeline-daily.py" secondly which will generate a new synthetic flower randomly based on our dataset via Modal to train the model. And we run the "iris-training-pipeline.ipynb" thirdly to download the features and labels, get the model with KNN and upload it. Finally, we run "iris-batch-inference-pipeline.py" to do prediction and upload our classification pictures, prediction table and confusion matrix. 

After all these, we complete a good pipeline and we can run "Iris UI app.py" which includes the Gradion UI on Hugging Face Space and we can input and feature number we want to predict. Also, we can download the logs we update and show them all together on Hugging face with "Iris Monitor UI app.py"



Task2: In Task2, our tasks are:

1) Write a feature pipeline notebook that registers the wine quality dataset as a Feature Group with Hopsworks.

2) Write a training pipeline that reads training data with a Feature View from Hopsworks, trains a regression or classifier model to predict if a wine’s quality. Register the model with Hopsworks.

3) Write a Gradio application that downloads your model from Hopsworks and provides a User Interface to allow users to enter or select feature values to predict the quality of a wine for the features you entered.

4) Write a synthetic wine generator function and write a new “daily” feature pipeline that runs once per day to add a new synthetic wine.

5) Write a batch inference pipeline to predict the quality of the new wine(s) added, and build a Gradio application to show the most recent wine quality prediction and outcome, and a confusion matrix with historical prediction performance. 

Pipeline picture:
![image](https://github.com/ZhihanX/Wine-quality/assets/114545801/02dfba17-33fd-4c85-a17d-3544593d01d7)

First, we use function value.count() to generate wine quality we will predict but the quantity of quality are concentrated in 5,6,7. So, when we generate confusion matrix, there are 5,6,7 exists only which is not good. So we try to 
