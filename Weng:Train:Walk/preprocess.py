import math
import argparse
import pandas as pd

def preprocess(acc_csv, gyr_csv, mag_csv, dest_fpath):

    df_acc = pd.read_csv(acc_csv)
    df_gyr = pd.read_csv(gyr_csv)
    df_mag = pd.read_csv(mag_csv)

    df_acc.dropna(inplace=True)
    df_gyr.dropna(inplace=True)
    df_mag.dropna(inplace=True)

    df_acc.drop(columns=["TS"], inplace=True)
    df_gyr.drop(columns=["TS"], inplace=True)
    df_mag.drop(columns=["TS"], inplace=True)

    df_acc['TI'] = df_acc['TI'].apply(lambda x: math.floor(x))
    df_gyr['TI'] = df_gyr['TI'].apply(lambda x: math.floor(x))
    df_mag['TI'] = df_mag['TI'].apply(lambda x: math.floor(x))

    merged_df = df_acc.merge(df_gyr.merge(df_mag, on=['TI']), on=['TI'])
    merged_df.columns = ['TI', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', \
                         'mag_x', 'mag_y', 'mag_z']

    merged_df = merged_df.groupby(['TI'], as_index=False).mean()

    merged_df.drop(columns=['TI'], inplace=True)

    merged_df.to_csv(dest_fpath, index=False)
    
    print(f"Merging completed, csv saved at {dest_fpath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Accelerator, Gyroscope and Magnetometer csv merger')
    parser.add_argument('acc_path', type=str)
    parser.add_argument('gyr_path', type=str)
    parser.add_argument('mag_path', type=str)
    parser.add_argument('dest_fpath', type=str)
    args = parser.parse_args()

    acc_csv = args.acc_path
    gyr_csv = args.gyr_path
    mag_csv = args.mag_path
    dest_fpath = args.dest_fpath

    preprocess(acc_csv, gyr_csv, mag_csv, dest_fpath)

