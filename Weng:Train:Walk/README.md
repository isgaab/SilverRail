# HackTrain 5.0: HackTrack
Claudia Sanchez, Gabby Colmenares Reverol, Nicole Jesse, Weng Chen

# Models folder
- Models are first trained then pickled. The folder should contain the following as main models:
gaby_knn.p
weng_random_forest.p 

Current app uses: 
1) gaby_knn.p:
- train_walking_model (~87% accuracy)
- bus_train_walking_model (~82% accuracy)

2) weng_random_forest.p:
- train_walking_model (~85% accuracy, 0.88-0.66 Precision-Recall for Walk, 0.84-0.95 Precision-Recall Train) 
- Very fast train time : normally less than a minute 
                 
#  How to run
1) To build the api endpoint as a docker image, in the root directory of project, please run `docker build -t <docker_name> .
2) Once the build is finished, run `docker run -it -p 3030:3030 <docker_name>` to launch the service.
3) Run `python rest_client.py` to send data csv file to API endpoints to retrieve the results. You are encouraged to look at rest_client.py to see how the data is sent.
4) The data csv file is expected to have been preprocess already and no mode of transport is given. Refer to `clean_merged_1_11_0.5.csv` for a quick view on the expected data structure.

# API endpoints structure
Current structure "http://localhost:3030/api/v1/predict/<model_name>"

Replace <model_name> with `model1` for Weng's model and `model2` for Gaby's model.

Note the api endpoint for weng's model expects a single merged CSV with columns ['TI', 'TS', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'mag_x', 'mag_y', 'mag_z'] .

# Output
Will be a JSON.
Weng's model generates 

output = [{'predVals' : array of results(0 or 1) }]

Gaby's model generates

output = [{f"val{x}":
          {"activity": 'walk' or 'train', 
           "accuracy": accuracy score, 
           "timestamp": TI
          } ...]


# Attention
1) Jupyter notebook has played a large role in this project for data analysis. Its is STRONGLY advised to have a look at weng_analysis_train_walk.ipynb to have some idea about the nature of data, the approaches taken to resolve issues and some advice on improving models.
2) The preprocess step which merge Accelerometer, Gyroscope and Magnetometer .csvs have to be done separately because large files will exhaust the memory. Note at this point it operates by assuming no mode of transport is given.


                  
