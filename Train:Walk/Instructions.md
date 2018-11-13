# Model Architecture

The model has already been trained, under the name of ""; however, the model can be recreated and retrained with new data, the dataset to be fed needs 
to have the following form (.csv):

|  TI          | TS               |  x           |  y              |  z              |  mode   |  dataset  |
| ------------ | ---------------  | ------------ | --------------- | --------------- | ------  | --------- |
| 16285.29835  | 30/10/2018 09:31 | 0.371414185  |  -0.397399902   |  -0.992050171   |  Train  |  1        |

Where:
  - TI: Time Index
  - TS: Timestamp
  - x: accelerometer data in x
  - y: accelerometer data in y
  - z: accelerometer data in z
  - mode: type of transport (walk or train)
  - dataset: identify different journeys, the model needs to identify between different datasets
  
  
# Model Testing - THIS IS THE FILE TO BE USED!

The file for testing must have the following form (.csv):


|  TI          | TS               |  x           |  y              |  z              |  mode   | 
| ------------ | ---------------  | ------------ | --------------- | --------------- | ------  |
| 16285.29835  | 30/10/2018 09:31 | 0.371414185  |  -0.397399902   |  -0.992050171   |  Train  | 


  Where:
  - TI: Time Index
  - TS: Timestamp
  - x: accelerometer data in x
  - y: accelerometer data in y
  - z: accelerometer data in z
  - mode: type of transport (walk or train)
