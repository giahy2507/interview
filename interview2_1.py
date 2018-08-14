from collections import defaultdict
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_validate, train_test_split, KFold
from copy import deepcopy

from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers


def read_data(data_file, label_file):
    """
    Load data function
    Params:
        data_file: data/trainingData.csv
        label_file: data/traininglabels.csv
    Returns:
        data (dictionary)
        categories (list(4)): categories information, each element is an dictionary[<category_id>:<frequency>]
    """
    data = []
    labels = []
    categories = [defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)]
    with open(label_file, mode="r") as f:
        for line in f:
            labels.append(line.strip())

    with open(data_file, mode="r") as f:
        for line_idx,line in enumerate(f):
            tokens = line.strip().split(",")
            session_id = tokens[0].strip()
            start_time = tokens[1].strip()
            end_time = tokens[2].strip()
            list_products_str = tokens[3].strip().split(";")
            list_products  = []
            for product_str in list_products_str:
                product_cats = product_str.rstrip("/").split("/")
                list_products.append(product_cats)
                for i in range(len(product_cats)):
                    categories[i][product_cats[i]] +=1
            data.append({"session_id":session_id,
                         "start_time": start_time,
                         "end_time": end_time,
                         "list_products": list_products,
                         "gender": labels[line_idx]})
    return data, categories


def time_statistic(data):
    """
    Male and female Statistic function for 24 hours
    Params:
        data: object from read_data()
    """
    time_male   = [0] * 25
    time_female = [0] * 25
    for row in data:
        if row["gender"] == "male":
            time_male[int(row["start_time"][-8:-6])] +=1
        else:
            time_female[int(row["start_time"][-8:-6])] += 1
    print(time_male)
    print(time_female)

def category1_statistic(data, categories, category_i = 0):
    """
    Male and female Statistic function for category_ith
    Params:
        data: object from read_data()
        categories: object from read_data()
        category_i: category_ith
    """
    category1 = categories[category_i]
    category_m = defaultdict(int)
    category_f = defaultdict(int)
    for row in data:
        for product in row["list_products"]:
            if row["gender"] == "male":
                category_m[product[category_i]]+=1
            else:
                category_f[product[category_i]] += 1
    print(category_m)
    print(category_f)
    for key in category1.keys():
        print(key, category_m[key], category_f[key])

def adjust_index(categories, min_df = 3):
    """
    Indexing product_id, category_id for vectorizing
    Params:
        categories: object from read_data()
        min_df: minimum frequency
    Returns:
        result: (list(4)): categories information, each element is an dictionary[<category_id>:<feature_id>]
        counter: number of features
    """
    result = deepcopy(categories)
    counter = 24 # the first 24 indexes for 24 hours
    for i in range(len(categories)):
        for key, value in categories[i].items():
            if value < min_df:
                continue
            result[i][key] = counter
            counter+=1
    return result, counter

def reverse_categories(categories):
    # reverse dictionary[<feature_id>:<category_id>]
    reversed_categories_flat = {}
    for category in categories:
        for key, value in category.items():
            reversed_categories_flat[value] = key
    return reversed_categories_flat

def category_items_statistic(data, items):
    candidate_categories = [item[1] for item in items]
    category_m = defaultdict(int)
    category_f = defaultdict(int)
    for row in data:
        for product in row["list_products"]:
            for product_cat in product:
                if product_cat in candidate_categories:
                    if row["gender"] == "male":
                        category_m[product_cat]+=1
                    else:
                        category_f[product_cat] += 1
    new_candidate_categories = [ (item[0],item[1], category_m[item[1]],category_f[item[1]]) for item in items]
    with open("data/mystatistic.txt", mode="w") as f:
        for item in new_candidate_categories:
            f.write(str(item)+"\n")

def feature_extraction(data, categories, min_df):
    """
    Feature extraction function.
    Each sample stand for an transaction.
    Params:
        data:  object from read_data()
        categories:  object from read_data()
        min_df: minimum frequency
    Return:
        X: feature matrix, shape (n_samples, n_features)
        Y: label vectors, shape (n_samples)
    """
    categories, no_fe = adjust_index(categories, min_df=min_df)
    X = np.zeros(shape=(len(data), no_fe), dtype=np.float32)
    Y = np.zeros(shape=(len(data),), dtype=np.int32)
    for i, row in enumerate(data):
        if row["gender"] == "male":
            Y[i] = 0
        else:
            Y[i] = 1

        # time_feature
        X[i][int(row["start_time"][-8:-6])] = 1

        for product in row["list_products"]:
            for product_cat_i, product_cat in enumerate(product):
                try:
                    feature_j = categories[product_cat_i][product_cat]
                    X[i][feature_j] = 1.0
                except:
                    continue
    return X, Y

# def top_kbest_feature_by_chi2(data,categories,X,Y):
#     chi2_result, pval = chi2(X, Y)
#     tops_features = []
#     reversed_categories_flat = reverse_categories(categories)
#
#     for i in range(24, len(chi2_result)):
#         heappush(tops_features, (chi2_result[i], reversed_categories_flat[i]))
#
#     x = nlargest(10000, tops_features)
#     category_items_statistic(data, x)

def y_to_onehot(y, n_class = 2):
    result = np.zeros(shape=(len(y), n_class), dtype=np.int32)
    for idx, label in enumerate(y):
        result[idx][label] = 1
    return result

def my_accuracy(y_true, y_pred):
    male_correct = 0.0
    male_total = 0.0
    fmale_correct = 0.0
    fmale_total = 0.0

    for label_true, label_pred in zip(y_true, y_pred):
        if label_true == 0:
            male_total +=1
            if label_pred == 0:
                male_correct+=1
        else:
            fmale_total+=1
            if label_pred == 1:
                fmale_correct+=1
    myacc = (male_correct/male_total + fmale_correct/fmale_total)/2.0
    return myacc

if __name__ == "__main__":

    results = defaultdict(list)
    # mindf_kbest_params = [(1,5000), (1,7500), (1,10000), (1,12500), (2,10000), (3,10000)]
    # mindf_kbest_params = [(1, 5000), (2, 10000), (3, 10000)]
    mindf_kbest_params = [(2, 10000)]

    for mindf_kbest in mindf_kbest_params:
        mindf, kbest = mindf_kbest

        data, categories = read_data("data/trainingData.csv", "data/trainingLabels.csv")
        X, y = feature_extraction(data, categories, min_df = mindf)
        y = y_to_onehot(y,2)

        kf = KFold(n_splits=10, shuffle=True)
        i_fold = 1

        for train_index, test_index in kf.split(X):
            print("----------", i_fold, "----------")

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # although I try to select K best features, the performance between them are comparable.
            # ch2 = SelectKBest(chi2, k=min(X_train.shape[1], kbest))
            # X_train = ch2.fit_transform(X_train, y_train)
            # X_test = ch2.transform(X_test)
            print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

            # 3-layer neural network model with 100 hidden units
            a = Input(shape=(X_train.shape[1],))
            b = Dense(100, activation="sigmoid", kernel_regularizer=regularizers.l2(0.001))(a)
            c = Dense(2,activation='softmax', kernel_regularizer=regularizers.l2(0.001))(b)
            model = Model(inputs=a, outputs=c)
            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            model.fit(X_train,
                      y_train,
                      epochs=10,
                      batch_size=16,
                      callbacks=[EarlyStopping(monitor="val_loss", patience=20, verbose=True),
                                 ModelCheckpoint("models/kr.model",verbose=True, save_best_only=True, monitor="val_loss")],
                      validation_data=(X_test, y_test), verbose=True)
            model.load_weights("models/kr.model")
            y_pred = model.predict(X_test)
            y_pred = np.array(np.argmax(y_pred, axis=-1))
            y_test = np.array(np.argmax(y_test, axis=-1))
            print("My acc", my_accuracy(y_test, y_pred))
            i_fold += 1

    print(results)
    # time_statistic(data)
    # category1_statistic(data, categories, 0)
    # categories = adjust_index(categories)