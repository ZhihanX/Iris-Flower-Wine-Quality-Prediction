import os
import modal

LOCAL=False

if LOCAL == False:
   stub = modal.Stub("wine_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("ID2223"))
   def f():
       g()


def generate_wine(quality_cls, fixed_acidity_min,fixed_acidity_max, volatile_acidity_min, volatile_acidity_max,
                  citric_acid_min, citric_acid_max, residual_sugar_min, residual_sugar_max, chlorides_min, chlorides_max, 
                  free_sulfur_dioxide_min, free_sulfur_dioxide_max, total_sulfur_dioxide_min, total_sulfur_dioxide_max, density_min,
                  density_max, ph_min, ph_max, sulphates_min, sulphates_max, alcohol_min, alcohol_max):
    """
    Returns a single wine as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df = pd.DataFrame({"type": [random.uniform(0,1)],
                       "fixed_acidity":[random.uniform(fixed_acidity_min,fixed_acidity_max)],
                       "volatile_acidity": [random.uniform(volatile_acidity_min,volatile_acidity_max)],
                       "citric_acid": [random.uniform(citric_acid_min,citric_acid_max)],
                       "residual_sugar": [random.uniform(residual_sugar_min,residual_sugar_max)],
                       "chlorides": [random.uniform(chlorides_min,chlorides_max)],
                       "free_sulfur_dioxide": [random.uniform(free_sulfur_dioxide_min,free_sulfur_dioxide_max)],
                       "total_sulfur_dioxide": [random.uniform(total_sulfur_dioxide_min,total_sulfur_dioxide_max)],
                       "density": [random.uniform(density_min,density_max)],
                       "ph": [random.uniform(ph_min,ph_max)],
                       "sulphates": [random.uniform(sulphates_min,sulphates_max)],
                       "alcohol": [random.uniform(alcohol_min,alcohol_max)],
                      })
    df['quality'] = quality_cls
    return df


def get_random_wine():
    """
    Returns a DataFrame containing one random wine flower
    """
    import pandas as pd
    import random

    good_df = generate_wine(0, 6.8, 8.57, 0.25, 0.63, 0.2125, 0.39500, 1.65, 7.8, 
                            0.03875,0.081,5.25,37.375,30.0,193.25,0.99375,0.998060,3.14,3.415,
                            0.4075,0.565,9.625,11.0)
    bad_df = generate_wine(1, 6.9, 7.4, 0.26, 0.36, 0.34, 0.45, 2, 4.2,0.021,0.032,27,31,
                           113,124,0.9898,0.99055,3.28,3.37,0.42,0.48,12.4,12.7)
    

    # randomly pick one of these 3 and write it to the featurestore
    pick_random = random.uniform(0,2)
    if pick_random >= 1:
        wine_df = good_df
        print("Good quality wine added")
    else:
        wine_df = bad_df
        print("Bad quality wine added")

    return wine_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login(project="ID2223_1")
    fs = project.get_feature_store()

    wine_df = get_random_wine()

    wine_fg = fs.get_feature_group(name="wine",version=1)
    wine_fg.insert(wine_df)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        #stub.deploy("wine_daily")
        with stub.run():
            f.remote()
