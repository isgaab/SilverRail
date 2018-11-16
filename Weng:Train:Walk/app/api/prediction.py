from flask import current_app, jsonify, request, abort, Response
from . import api

import io
import math
import pandas as pd
import numpy as np
from time import time

# Hard coded n nearest neighbours, can be changed to customisable
n_neighbors = 25


#####################################################################################
# Gaby's function
def half_int(x):
    val=0.5 * math.ceil(2.0 * x)
    return val

#Cleaning function
def cleaning(df_acc):
    # Rename df_acc column, 'train' -> 'mode'
    df_acc.columns = ['TI', 'TS', 'x', 'y', 'z', 'mode', 'dataset']

    # Drop TS column and replace transportation mode into integer
    df_acc = df_acc.replace({'mode': {'Train': 1, 'Walking': 0}})

    # Convert TI value to integer
    df_acc['TI'] = df_acc['TI'].apply(lambda x: half_int(x))

    # Group by and get mean by TI as key
    ndf_acc = df_acc.groupby(['TI', 'dataset'], as_index=False)[['x', 'y', 'z', 'mode']].mean()

    # Clean filtered data
    ndf_acc.columns = ['TI', 'dataset', 'acc_x', 'acc_y', 'acc_z', 'mode']
    
    return ndf_acc

##########################################################################################
# Weng's function
def preprocess(acc_file, gyr_file, mag_file):
    df_acc = pd.read_csv(acc_file)
    df_gyr = pd.read_csv(gyr_file)
    df_mag = pd.read_csv(mag_file)

    df_acc = df_acc[df_acc['mode'] != 'Bus']
    df_gyr= df_gyr[df_gyr['mode'] != 'Bus']
    df_mag = df_mag[df_mag['mode'] != 'Bus']

    df_acc.dropna(inplace=True)
    df_gyr.dropna(inplace=True)
    df_mag.dropna(inplace=True)

    df_acc = df_acc.drop(columns=["TS"]).replace({'mode': {'Train': 1, 'Walking': 0}})
    df_gyr = df_gyr.drop(columns=["TS"]).replace({'mode': {'Train': 1, 'Walking': 0}})
    df_mag = df_mag.drop(columns=["TS"]).replace({'mode': {'Train': 1, 'Walking': 0}})

    df_acc['TI'] = df_acc['TI'].apply(lambda x: math.floor(x))
    df_gyr['TI'] = df_gyr['TI'].apply(lambda x: math.floor(x))
    df_mag['TI'] = df_mag['TI'].apply(lambda x: math.floor(x))

    merged_df = df_acc.merge(df_gyr.merge(df_mag, on=['TI']), on=['TI'])
    merged_df.columns = ['TI', 'acc_x', 'acc_y', 'acc_z', 'acc_mode', 'gyr_x', 'gyr_y', 'gyr_z', 'gyr_mode', \
                         'mag_x', 'mag_y', 'mag_z', 'mag_mode']
    merged_df['mode'] = merged_df[['acc_mode', 'gyr_mode', 'mag_mode']].mean(axis=1)

    merged_df = merged_df.groupby(['TI'], as_index=False).mean()

    merged_df['mode'] = merged_df[['acc_mode', 'gyr_mode', 'mag_mode']].mean(axis=1)
    merged_df['mode'] = merged_df['mode'].apply(lambda x: conv(x))

    merged_df.drop(columns=['TI', 'acc_mode', 'gyr_mode', 'mag_mode'], inplace=True)
    
    return merged_df


def gen_feat(df):

    df['acc_norm'] = np.linalg.norm(df[['acc_x','acc_y','acc_z']].values,axis=1)
    df['gyr_norm'] = np.linalg.norm(df[['gyr_x','gyr_y','gyr_z']].values,axis=1)
    df['mag_norm'] = np.linalg.norm(df[['mag_x','mag_y','mag_z']].values,axis=1)
    
    df['acc_mean'] = df[['acc_x','acc_y','acc_z']].mean(axis=1)
    df['gyr_mean'] = df[['gyr_x','gyr_y','gyr_z']].mean(axis=1)
    df['mag_mean'] = df[['mag_x','mag_y','mag_z']].mean(axis=1)
    
    df['acc_median'] = df[['acc_x','acc_y','acc_z']].median(axis=1)
    df['gyr_median'] = df[['gyr_x','gyr_y','gyr_z']].median(axis=1)
    df['mag_median'] = df[['mag_x','mag_y','mag_z']].median(axis=1)
    
    df['acc_var'] = df[['acc_x','acc_y','acc_z']].var(axis=1)
    df['gyr_var'] = df[['gyr_x','gyr_y','gyr_z']].var(axis=1)
    df['mag_var'] = df[['mag_x','mag_y','mag_z']].var(axis=1)
    
    return df

##################################################################################
@api.route("/predict/<model_name>", methods=['POST'])
def predict(model_name):
    # Expects user data i.e gyro, accelero etc
    
    try:
        if model_name == "model1":
            user_data = request.files
            merged_df = pd.read_csv(user_data['data'])
            merged_df.drop(columns=['mode'], inplace=True)
            # acc_data = user_data['acc']
            # gyr_data = user_data['gyr']
            # mag_data = user_data['mag']
            
            # merged_df = preprocess(acc_data, gyr_data, mag_data)
            feat_df = gen_feat(merged_df)

            
            predVals = current_app.mlmodels.process(model_name, feat_df.values)
            

            output = [{'predVals' : predVals.tolist()}]
            

        elif model_name == "model2":
            user_data = request.files
            acc_data = user_data['acc']
            #df_acc = pd.read_csv(io.BytesIO(user_data), encoding="utf-8", sep=",")
            df_acc = pd.read_csv(acc_data)
            df = cleaning(df_acc)
            
            # Magnitude dataframe
            df['acc_magnitude']=(df['acc_x']**2+df['acc_y']**2+df['acc_z']**2)**(1/2.0)
            
            # Window size = ws, window delay = wd
            ws, wd = 10, 15
            X_new = [df['acc_magnitude'].values[x:x+ws] for x in range(0,len(df['acc_magnitude'])-ws+1, wd) if (df.iloc[x]['mode'] == df.iloc[x+ws-1]['mode'])]
            X_new = np.asarray(X_new)

            Y_new = [df['mode'].values[x] for x in range(0,len(df['mode'])-ws+1, wd) if df.iloc[x]['mode'] == df.iloc[x+ws-1]['mode']]
            Y_new = np.asarray(Y_new)

            times = [df['TI'].values[x] for x in range(0,len(df['TI'])-ws+1, wd) if df.iloc[x]['mode'] == df.iloc[x+ws-1]['mode']]

            times_ref = df_acc
            times_ref['TI'] = df['TI'].apply(lambda x: half_int(x))
            times_ref = times_ref.loc[times_ref['TI'].isin(times)]
            times_ref = times_ref.drop_duplicates(subset='TI', keep="last")['TS']

            # Modifying data
            predVals = current_app.mlmodels.process(model_name, X_new)

            activity = {0: "walk", 1: "train"}

            act, acc = predVals

            output = [{f"val{x}":
               {"activity": activity[act[x]], "accuracy": acc[x][int(act[x])], "timestamp": times_ref.iloc[x]}
               } for x in range(len(act))]

        current_app.logger.warn(output)
        current_app.logger.warn("====================================+")

    except Exception as e:
        current_app.logger.error(f"Caught an error : {e}", exc_info=True)
        abort(Response("Internal system error, please try again", 500))

    return jsonify(output=output)

