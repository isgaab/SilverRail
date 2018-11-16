train_df = pd.read_csv("../data/final_data/clean_merged_1_11_0.5.csv")
test_df = pd.read_csv("../data/final_data/merged_df12.csv")

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


# Rearranging the 'model' column for convenience
train_df = gen_feat(train_df)

tmode = train_df['mode']
train_df.drop(columns=['mode'], inplace=True)
train_df['mode'] = tmode

# Repeat for test data
test_df = gen_feat(test_df)

tmode = test_df['mode']
test_df.drop(columns=['mode'], inplace=True)
test_df['mode'] = tmode


# Shuffle the train data to ensure data does not depend on sequence
train_data_shuf = train_df.values
np.random.shuffle(train_data_shuf)

X_train_shuf, Y_train_shuf = train_data_shuf[:, :-1], train_data_shuf[:, -1]

# Shuffle the test data
test_data_shuf = test_df.values
np.random.shuffle(test_data_shuf)

X_test_shuf, Y_test_shuf = test_data_shuf[:, :-1], test_data_shuf[:, -1]

depths = (5, 10, 20)
n_estimators = (10, 50, 100, 150, 200)

target_names = ('Walk', 'Train')

rclfs = []

for d in depths:
    for n in n_estimators:
        print(f"Tree depth : {d}. n_estimators: {n}.")
        rclf = RandomForestClassifier(n_estimators=n, max_depth=d)
        rclf.fit(X_train_shuf, Y_train_shuf)
        Y_pred = rclf.predict(X_test_shuf)
        print((Y_pred == Y_test_shuf).mean())
        average_precision = average_precision_score(Y_test_shuf, Y_pred)
        print(average_precision)
        print(classification_report(Y_test_shuf, Y_pred, target_names=target_names))
        rclfs.append(rclf)


rclf = RandomForestClassifier(n_estimators=150, max_depth=5)
rclf.fit(X_train_shuf, Y_train_shuf)
pickle.dump(rclf, open("../models/weng_random_forest.p", "wb"))

Y_pred = rclf.predict(X_test_shuf)
print((Y_pred == Y_test_shuf).mean())
average_precision = average_precision_score(Y_test_shuf, Y_pred)
print(average_precision)
print(classification_report(Y_test_shuf, Y_pred, target_names=target_names))
