from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Activation, Input, concatenate
from keras.models import Model
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.exceptions import NotFittedError
from subprocess import check_output
from math import ceil
import pandas as pd
from csv import reader
from multiprocessing import Process, Queue

data_dir = "data"
processed_dir = "processed"

with open(f"{data_dir}/genres.txt", "r") as file:
    num_genres = len(file.read().split(";"))

means = pd.read_csv("processed/means.csv", header=None).values
sdevs = pd.read_csv("processed/sdevs.csv", header=None).values


def line_count(file):
    return int(check_output(f"wc -l {file}", shell=True).split()[0])


def get_num_samples(datasets, mode="train"):
    n = 0
    for dataset in datasets:
        n += line_count(f"{processed_dir}/{mode}/{dataset}.genres.csv")
    return n


def accumulate_batches(datasets, mode, queue):
    rows_to_read = 20000
    for dataset in datasets:
        x_path = f"{processed_dir}/{mode}/{dataset}.csv"
        y_path = f"{processed_dir}/{mode}/{dataset}.genres.csv"
        with open(y_path, "r") as file:
            rows = reader(file)
            y = []
            row_number = 0
            for row in rows:
                row_number += 1
                y.append([int(value) for value in row[1:]])
                if row_number % rows_to_read == 0:
                    try:
                        y = y_binarizer.transform(y)
                    except NotFittedError:
                        y = y_binarizer.fit_transform(y)
                    x = pd.read_csv(x_path, skiprows=row_number, nrows=rows_to_read, index_col=0, header=None)
                    x.dropna(inplace=True)
                    x = x.values
                    x -= means
                    x /= sdevs
                    queue.put((x, y))
                    y = []


def batch_generator(datasets, mode="train"):
    queue = Queue()
    process = Process(target=accumulate_batches, args=(datasets, mode, queue))
    process.start()
    while True:
        x, y = queue.get()
        for i in range(int(len(y) / batch_size)):
            yield x[(i * batch_size):batch_size], y[(i * batch_size):batch_size]


y_binarizer = MultiLabelBinarizer(classes=range(num_genres))
datasets = ["discogs"]
input_dim = 2670
init = "uniform"
dropout_prob = 0.5
n_units = 256
epochs = 100
batch_size = 32


def get_embedding_model(finetune=False, input1=None):
    if finetune:
        if input1 is None:
            raise ValueError
        x = Dropout(dropout_prob, trainable=False)(input1)
        output = Dense(n_units, kernel_initializer=init, activation='relu', kernel_regularizer="l2", trainable=False)(x)
    else:
        input1 = Input(shape=(input_dim,))
        x = Dropout(dropout_prob)(input1)
        x = Dense(n_units, kernel_initializer=init, activation='relu')(x)
        x = Dropout(dropout_prob)(x)
        output = Dense(num_genres, kernel_initializer="uniform", activation="sigmoid")(x)
    return input1, output


def get_fusion_model():
    input1 = Input(shape=(input_dim,))
    embedding_models = []
    for i in range(4):
        embedding_models.append(get_embedding_model(finetune=True, input1=input1)[1])
    x = concatenate(embedding_models)
    x = Dropout(dropout_prob)(x)
    output = Dense(num_genres, kernel_initializer="uniform", activation="sigmoid")(x)
    return input1, output


def train():
    in1, out = get_embedding_model()
    model = Model(inputs=in1, outputs=out)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["mean_squared_error"])
    print(model.summary())
    early_stopping = EarlyStopping(monitor="val_loss", patience=4)
    model.fit_generator(batch_generator(datasets),
              steps_per_epoch=ceil(get_num_samples(datasets, mode="train") / batch_size),
              epochs=epochs,
              callbacks=[early_stopping],
              validation_data=batch_generator(datasets, mode="validation"),
              validation_steps=ceil(get_num_samples(datasets, mode="validation") / batch_size))
    model.save("model.h5")


def main():
    train()


if __name__ == "__main__":
    main()
