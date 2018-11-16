import base64
import pprint 
import requests
import json
import numpy as np 

pp = pprint.PrettyPrinter(indent=4)

content_type= "application/json"
headers = {'content-type': content_type}

#input_data = [[-1.27947235e-01, -6.27358246e-01, 7.76403809e-01, -3.59243529e+02, -2.88740537e+02, 1.00635567e+00]]
#data = json.dumps(input_data)

#data = json.dumps(data)


#files = {'acc': open('Accelerometer.csv', 'rb'),\
#        'gyr': open('Gyroscope.csv', 'rb'),\
#        'mag': open('Magnetometer.csv', 'rb')}

files = {'data': open('merged_df12.csv', 'rb')}

json_response = requests.post("http://localhost:3030/api/v1/predict/model1", files=files)

print("----------JSON_response-----------")
print(json_response.status_code)
print(json_response.status_code == 200)
print("----------Nearest Neighbor Path + Distance----------")
response = json.loads(json_response.text)
pp.pprint(response)
