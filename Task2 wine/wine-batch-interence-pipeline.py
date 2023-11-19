
import os
import modal
    
LOCAL=False

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","dataframe-image","scikit-learn==1.1.1"])
   #"sklearn==1.1.1",
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=2)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")
    
    feature_view = fs.get_feature_view(name="wine_bin_classify_269", version=1)
    batch_data = feature_view.get_batch_data()
    
    y_pred = model.predict(batch_data)
    #print(y_pred)
    offset = 500 # Bad for 100 Good for 500
    wine = y_pred[y_pred.size-offset]
    if(wine==float(1)):
        wine = "Good"
        wine_url = "https://i.ytimg.com/vi/9wFm7wTJ7JU/maxresdefault.jpg"
    else :
        wine = "Bad"
        wine_url = "https://media.istockphoto.com/id/117068556/sv/foto/bad-wine.jpg?s=2048x2048&w=is&k=20&c=wLOisv5qh9N8bp8AISRo1yP2nOjq_ouvt4sWeZ11yy0="
    # wine_url = "https://raw.githubusercontent.com/featurestoreorg/serverless-ml-course/main/src/01-module/assets/" + wine + ".png"
    print("Wine quality predicted: " + wine)
    img = Image.open(requests.get(wine_url, stream=True).raw)            
    img.save("./latest_wine.png")
    dataset_api = project.get_dataset_api()    
    dataset_api.upload("./latest_wine.png", "Resources/images", overwrite=True)
   
    wine_fg = fs.get_feature_group(name="wine_bin_classify_269", version=1)
    df = wine_fg.read(read_options={"use_hive": True}) 
    #print(df)
    label = df.iloc[-offset]["quality"]
    if(label==float(1)):
        label = "Good"
        label_url = "https://i.ytimg.com/vi/9wFm7wTJ7JU/maxresdefault.jpg"

    else :
        label = "Bad"
        label_url = "https://media.istockphoto.com/id/117068556/sv/foto/bad-wine.jpg?s=2048x2048&w=is&k=20&c=wLOisv5qh9N8bp8AISRo1yP2nOjq_ouvt4sWeZ11yy0="

    # label_url = "https://raw.githubusercontent.com/featurestoreorg/serverless-ml-course/main/src/01-module/assets/" + label + ".png"
    print("Wine quality: " + label)
    img = Image.open(requests.get(label_url, stream=True).raw)            
    img.save("./actual_wine.png")
    dataset_api.upload("./actual_wine.png", "Resources/images", overwrite=True)
    
    monitor_fg = fs.get_or_create_feature_group(name="wine_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Wine quality Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [wine],
        'label': [label],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read(read_options={"use_hive": True})
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])


    df_recent = history_df.tail(4)
    dfi.export(df_recent, './df_recent_wine.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent_wine.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our iris_predictions feature group has examples of all 3 iris flowers
    print("Number of different wine quality predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 2:
        results = confusion_matrix(labels, predictions)
    
        df_cm = pd.DataFrame(results, ['True Bad', 'True Good'],
                             ['Pred Bad', 'Pred Bad'])
    
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix_wine.png")
        dataset_api.upload("./confusion_matrix_wine.png", "Resources/images", overwrite=True)
    else:
        print("You need 2 different wine quality predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 2 different wine quality predictions") 

if __name__ == "__main__":
    if LOCAL == True :
        print("local")
    else:
        with stub.run():
            f.remote()
