# -*- coding: utf-8 -*-
import pandas as pd
# import built-in module

# import third-party modules
import tensorflow as tf
import numpy as np

# import your own module
from brainmepnas import Dataset, AccuracyMetrics
from brainmepnas.tf_utils import generate_tflite_model


PATIENT = 3


def get_epidenet_1ch_og() -> tf.keras.Model:
    epidenet_1ch_og = tf.keras.Sequential()
    epidenet_1ch_og.add(
        tf.keras.layers.InputLayer((768, 1, 1)))
    epidenet_1ch_og.add(
        tf.keras.layers.Conv2D(4, kernel_size=(4, 1), padding="same"))
    epidenet_1ch_og.add(tf.keras.layers.BatchNormalization())
    epidenet_1ch_og.add(tf.keras.layers.Activation("relu"))
    epidenet_1ch_og.add(tf.keras.layers.MaxPooling2D(pool_size=(8, 1)))
    epidenet_1ch_og.add(
        tf.keras.layers.Conv2D(16, kernel_size=(16, 1), padding="same"))
    epidenet_1ch_og.add(tf.keras.layers.BatchNormalization())
    epidenet_1ch_og.add(tf.keras.layers.Activation("relu"))
    epidenet_1ch_og.add(tf.keras.layers.MaxPooling2D(pool_size=(4, 1)))
    epidenet_1ch_og.add(
        tf.keras.layers.Conv2D(16, kernel_size=(8, 1), padding="same"))
    epidenet_1ch_og.add(tf.keras.layers.BatchNormalization())
    epidenet_1ch_og.add(tf.keras.layers.Activation("relu"))
    epidenet_1ch_og.add(tf.keras.layers.AveragePooling2D(pool_size=(8, 1)))
    epidenet_1ch_og.add(tf.keras.layers.Flatten())
    epidenet_1ch_og.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    # epidenet_1ch_og.summary()
    return epidenet_1ch_og


def get_epidenet_1ch_v3_2() -> tf.keras.Model:
    epidenet_1ch_v3_2 = tf.keras.Sequential()
    epidenet_1ch_v3_2.add(
        tf.keras.layers.InputLayer((768, 1, 1)))
    epidenet_1ch_v3_2.add(tf.keras.layers.ZeroPadding2D(padding=(2, 0)))
    epidenet_1ch_v3_2.add(tf.keras.layers.Conv2D(4, kernel_size=(4, 1)))
    epidenet_1ch_v3_2.add(tf.keras.layers.BatchNormalization())
    epidenet_1ch_v3_2.add(tf.keras.layers.Activation("relu"))
    epidenet_1ch_v3_2.add(tf.keras.layers.MaxPooling2D(pool_size=(8, 1)))

    epidenet_1ch_v3_2.add(tf.keras.layers.ZeroPadding2D(padding=(2, 0)))
    epidenet_1ch_v3_2.add(tf.keras.layers.Conv2D(16, kernel_size=(5, 1)))
    epidenet_1ch_v3_2.add(tf.keras.layers.BatchNormalization())
    epidenet_1ch_v3_2.add(tf.keras.layers.Activation("relu"))
    epidenet_1ch_v3_2.add(tf.keras.layers.ZeroPadding2D(padding=(2, 0)))
    epidenet_1ch_v3_2.add(tf.keras.layers.Conv2D(16, kernel_size=(5, 1)))
    epidenet_1ch_v3_2.add(tf.keras.layers.BatchNormalization())
    epidenet_1ch_v3_2.add(tf.keras.layers.Activation("relu"))
    epidenet_1ch_v3_2.add(tf.keras.layers.ZeroPadding2D(padding=(2, 0)))
    epidenet_1ch_v3_2.add(tf.keras.layers.Conv2D(16, kernel_size=(5, 1)))
    epidenet_1ch_v3_2.add(tf.keras.layers.BatchNormalization())
    epidenet_1ch_v3_2.add(tf.keras.layers.Activation("relu"))
    epidenet_1ch_v3_2.add(tf.keras.layers.ZeroPadding2D(padding=(2, 0)))
    epidenet_1ch_v3_2.add(tf.keras.layers.Conv2D(16, kernel_size=(4, 1)))
    epidenet_1ch_v3_2.add(tf.keras.layers.BatchNormalization())
    epidenet_1ch_v3_2.add(tf.keras.layers.Activation("relu"))
    epidenet_1ch_v3_2.add(tf.keras.layers.MaxPooling2D(pool_size=(4, 1)))

    epidenet_1ch_v3_2.add(tf.keras.layers.ZeroPadding2D(padding=(2, 0)))
    epidenet_1ch_v3_2.add(tf.keras.layers.Conv2D(16, kernel_size=(8, 1)))
    epidenet_1ch_v3_2.add(tf.keras.layers.BatchNormalization())
    epidenet_1ch_v3_2.add(tf.keras.layers.Activation("relu"))
    epidenet_1ch_v3_2.add(tf.keras.layers.MaxPooling2D(pool_size=(4, 1)))

    epidenet_1ch_v3_2.add(
        tf.keras.layers.AveragePooling2D(pool_size=(7, 1), strides=(1, 1)))
    epidenet_1ch_v3_2.add(tf.keras.layers.Flatten())
    epidenet_1ch_v3_2.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    epidenet_1ch_v3_2.summary()
    return epidenet_1ch_v3_2


def get_epidenet_4ch() -> tf.keras.Model:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer((768, 4, 1)))
    model.add(tf.keras.layers.Conv2D(4, kernel_size=(4, 1), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(8, 1)))
    model.add(tf.keras.layers.Conv2D(16, kernel_size=(16, 1), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(4, 1)))
    model.add(tf.keras.layers.Conv2D(16, kernel_size=(8, 1), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(4, 1)))
    model.add(tf.keras.layers.Conv2D(16, kernel_size=(1, 12), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 3)))
    model.add(tf.keras.layers.Conv2D(16, kernel_size=(1, 6), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(6, 1)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation("sigmoid"))
    # model.summary()

    return model


def predict_tflite(model_path, X) -> np.ndarray:
    interpreter_og = tf.lite.Interpreter(model_path)
    interpreter_og.allocate_tensors()
    input_details = interpreter_og.get_input_details()
    output_details = interpreter_og.get_output_details()
    y = np.empty((X.shape[0], 1))

    X = np.expand_dims(X.astype(np.float32), axis=3)
    for i, x in enumerate(X):
        interpreter_og.set_tensor(input_details[0]["index"], [x])
        interpreter_og.invoke()
        y[i] = interpreter_og.get_tensor(output_details[0]["index"])

    return y


if __name__ == '__main__':
    dataset = Dataset(r"../data/chbmit_768samples")

    results = []

    for fold, (train_records, test_records) in enumerate(dataset.split_leave_one_record_out(str(PATIENT))):
        # Load data (4 channels, 768 samples)
        train_X, train_y = dataset.get_data(train_records, set="train",
                                      shuffle=True, shuffle_seed=42)
        test_X, test_y = dataset.get_data(test_records, set="test", shuffle=False)

        # Create data with 1 channel for the models that require it.
        train_X_1ch = np.empty((train_X.shape[0]*train_X.shape[2], train_X.shape[1], 1))
        train_y_1ch = np.empty((train_X_1ch.shape[0], 1))
        test_X_1ch = np.empty((test_X.shape[0] * test_X.shape[2], test_X.shape[1], 1))
        test_y_1ch = np.empty((test_X_1ch.shape[0], 1))

        for i, val in enumerate(train_X):
            train_X_1ch[(i * 4)] = np.expand_dims(val[:, 0], -1)
            train_X_1ch[(i * 4) + 1] = np.expand_dims(val[:, 1], -1)
            train_X_1ch[(i * 4) + 2] = np.expand_dims(val[:, 2], -1)
            train_X_1ch[(i * 4) + 3] = np.expand_dims(val[:, 3], -1)

        for i, val in enumerate(train_y):
            train_y_1ch[(i * 4)] = val
            train_y_1ch[(i * 4) + 1] = val
            train_y_1ch[(i * 4) + 2] = val
            train_y_1ch[(i * 4) + 3] = val

        for i, val in enumerate(test_X):
            test_X_1ch[(i * 4)] = np.expand_dims(val[:, 0], -1)
            test_X_1ch[(i * 4) + 1] = np.expand_dims(val[:, 1], -1)
            test_X_1ch[(i * 4) + 2] = np.expand_dims(val[:, 2], -1)
            test_X_1ch[(i * 4) + 3] = np.expand_dims(val[:, 3], -1)

        for i, val in enumerate(test_y):
            test_y_1ch[(i * 4)] = val
            test_y_1ch[(i * 4) + 1] = val
            test_y_1ch[(i * 4) + 2] = val
            test_y_1ch[(i * 4) + 3] = val

        # Compile the models
        monitoring_metrics = [tf.keras.metrics.AUC(num_thresholds=25,
                                                   curve='PR',
                                                   name="auc_pr")]

        epidenet_1ch_og = get_epidenet_1ch_og()
        epidenet_1ch_v3_2 = get_epidenet_1ch_og()
        epidenet_4ch = get_epidenet_4ch()

        epidenet_1ch_og.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1 * 10 ** -4,
                                               beta_1=0.9, beta_2=0.999),
            loss="binary_crossentropy",
            metrics=monitoring_metrics)
        epidenet_1ch_v3_2.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1 * 10 ** -4,
                                               beta_1=0.9, beta_2=0.999),
            loss="binary_crossentropy",
            metrics=monitoring_metrics)
        epidenet_4ch.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1 * 10 ** -4,
                                               beta_1=0.9, beta_2=0.999),
            loss="binary_crossentropy",
            metrics=monitoring_metrics)

        # Train models

        # 1ch, og
        epidenet_1ch_og.fit(train_X_1ch, train_y_1ch,
                            validation_split=0.2,
                            epochs=1000, batch_size=256,
                            verbose=1,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                      patience=10,
                                                      mode="min",
                                                      start_from_epoch=10)],
                            use_multiprocessing=False, shuffle=False)

        epidenet_1ch_og_tflite = generate_tflite_model(epidenet_1ch_og,
                                                       input_format="float",
                                                       output_format="float",
                                                       representative_input=train_X_1ch)
        epidenet_1ch_og_tflite_path = f"epidenet_1ch_og_patient_{PATIENT}_fold_{fold}.tflite"
        with open(epidenet_1ch_og_tflite_path, "wb") as file:
            file.write(epidenet_1ch_og_tflite)
        train_y_1ch_pred = predict_tflite(epidenet_1ch_og_tflite_path, train_X_1ch)

        train_y_1ch_pred_mean = np.empty((int(train_y_1ch_pred.shape[0] / 4), train_y_1ch_pred.shape[1]))
        for i in range(train_y_1ch_pred_mean.shape[0]):
            train_y_1ch_pred_mean[i] = np.mean([train_y_1ch_pred[(i * 4)],
                                            train_y_1ch_pred[(i * 4) + 1],
                                            train_y_1ch_pred[(i * 4) + 2],
                                            train_y_1ch_pred[(i * 4) + 3]])

        threshold_1ch_og = AccuracyMetrics(train_y_1ch.flatten(),
                                           train_y_1ch_pred.flatten(), 3, 3,
                                           threshold="max_sample_f_score").threshold
        threshold_1ch_og_mean = AccuracyMetrics(train_y.flatten(),
                                                train_y_1ch_pred_mean.flatten(), 3, 3,
                                           threshold="max_sample_f_score").threshold

        # 1ch, v3.2
        epidenet_1ch_v3_2.fit(train_X_1ch, train_y_1ch,
                            validation_split=0.2,
                            epochs=1000, batch_size=256,
                            verbose=1,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                      patience=10,
                                                      mode="min",
                                                      start_from_epoch=10)],
                            use_multiprocessing=False, shuffle=False)

        epidenet_1ch_v3_2_tflite = generate_tflite_model(epidenet_1ch_v3_2,
                                                       input_format="float",
                                                       output_format="float",
                                                       representative_input=train_X_1ch)
        epidenet_1ch_v3_2_tflite_path = f"epidenet_1ch_v3_2_patient_{PATIENT}_fold_{fold}.tflite"
        with open(epidenet_1ch_v3_2_tflite_path, "wb") as file:
            file.write(epidenet_1ch_v3_2_tflite)
        train_y_1ch_pred = predict_tflite(epidenet_1ch_v3_2_tflite_path, train_X_1ch)

        train_y_1ch_pred = epidenet_1ch_v3_2.predict(train_X_1ch)
        train_y_1ch_pred_mean = np.empty(
            (int(train_y_1ch_pred.shape[0] / 4), train_y_1ch_pred.shape[1]))
        for i in range(train_y_1ch_pred_mean.shape[0]):
            train_y_1ch_pred_mean[i] = np.mean([train_y_1ch_pred[(i * 4)],
                                                train_y_1ch_pred[(i * 4) + 1],
                                                train_y_1ch_pred[(i * 4) + 2],
                                                train_y_1ch_pred[(i * 4) + 3]])

        threshold_1ch_v3_2 = AccuracyMetrics(train_y_1ch.flatten(),
                                           train_y_1ch_pred.flatten(), 3, 3,
                                           threshold="max_sample_f_score").threshold
        threshold_1ch_v3_2_mean = AccuracyMetrics(train_y.flatten(),
                                                train_y_1ch_pred_mean.flatten(),
                                                3, 3,
                                                threshold="max_sample_f_score").threshold

        # 4ch
        epidenet_4ch.fit(train_X, train_y,
                              validation_split=0.2,
                              epochs=1000, batch_size=256,
                              verbose=1,
                              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                      patience=10,
                                                      mode="min",
                                                      start_from_epoch=10)],
                              use_multiprocessing=False, shuffle=False)

        epidenet_4ch_tflite = generate_tflite_model(epidenet_4ch,
                                                         input_format="float",
                                                         output_format="float",
                                                         representative_input=train_X)
        epidenet_4ch_tflite_path = f"epidenet_4ch_patient_{PATIENT}_fold_{fold}.tflite"
        with open(epidenet_4ch_tflite_path, "wb") as file:
            file.write(epidenet_4ch_tflite)
        train_y_pred = predict_tflite(epidenet_4ch_tflite_path, train_X)

        train_y_pred = epidenet_4ch.predict(train_X)

        threshold_4ch = AccuracyMetrics(train_y.flatten(),
                                             train_y_pred.flatten(), 3, 3,
                                             threshold="max_sample_f_score").threshold

        # Test models

        # 1ch, og
        test_y_1ch_pred = predict_tflite(epidenet_1ch_og_tflite_path, test_X_1ch)

        test_y_1ch_pred_mean = np.empty((int(test_y_1ch_pred.shape[0] / 4), test_y_1ch_pred.shape[1]))
        for i in range(test_y_1ch_pred_mean.shape[0]):
            test_y_1ch_pred_mean[i] = np.mean([test_y_1ch_pred[(i * 4)],
                                            test_y_1ch_pred[(i * 4) + 1],
                                            test_y_1ch_pred[(i * 4) + 2],
                                            test_y_1ch_pred[(i * 4) + 3]])

        am_1ch_og = AccuracyMetrics(test_y_1ch.flatten(),
                                    test_y_1ch_pred.flatten(), 3, 3,
                                    threshold=threshold_1ch_og)
        d = am_1ch_og.as_dict()
        d["model"] = "epidenet_1ch_og"
        d["patient"] = PATIENT
        d["fold"] = fold
        results.append(d)

        am_1ch_og_mean = AccuracyMetrics(test_y.flatten(),
                                    test_y_1ch_pred_mean.flatten(), 3, 3,
                                    threshold=threshold_1ch_og_mean)
        d = am_1ch_og_mean.as_dict()
        d["model"] = "epidenet_1ch_og_mean"
        d["patient"] = PATIENT
        d["fold"] = fold
        results.append(d)

        # 1ch, v3.2
        test_y_1ch_v3_2_pred = predict_tflite(epidenet_1ch_v3_2_tflite_path,
                                         test_X_1ch)

        test_y_1ch_v3_2_pred_mean = np.empty(
            (int(test_y_1ch_pred.shape[0] / 4), test_y_1ch_pred.shape[1]))
        for i in range(test_y_1ch_v3_2_pred_mean.shape[0]):
            test_y_1ch_v3_2_pred_mean[i] = np.mean([test_y_1ch_v3_2_pred[(i * 4)],
                                               test_y_1ch_v3_2_pred[(i * 4) + 1],
                                               test_y_1ch_v3_2_pred[(i * 4) + 2],
                                               test_y_1ch_v3_2_pred[(i * 4) + 3]])

        am_1ch_v3_2 = AccuracyMetrics(test_y_1ch.flatten(),
                                    test_y_1ch_v3_2_pred.flatten(), 3, 3,
                                    threshold=threshold_1ch_v3_2)
        d = am_1ch_v3_2.as_dict()
        d["model"] = "epidenet_1ch_v3_2"
        d["patient"] = PATIENT
        d["fold"] = fold
        results.append(d)

        am_1ch_v3_2_mean = AccuracyMetrics(test_y.flatten(),
                                         test_y_1ch_v3_2_pred_mean.flatten(), 3, 3,
                                         threshold=threshold_1ch_v3_2_mean)
        d = am_1ch_v3_2_mean.as_dict()
        d["model"] = "epidenet_1ch_v3_2_mean"
        d["patient"] = PATIENT
        d["fold"] = fold
        results.append(d)

        # 4ch
        test_y_pred = predict_tflite(epidenet_4ch_tflite_path,
                                         test_X)

        am_4ch = AccuracyMetrics(test_y.flatten(),
                                    test_y_pred.flatten(), 3, 3,
                                    threshold=threshold_4ch)
        d = am_4ch.as_dict()
        d["model"] = "epidenet_4ch"
        d["patient"] = PATIENT
        d["fold"] = fold
        results.append(d)

        df = pd.DataFrame.from_records(results)
        df.to_csv(f"patient_{PATIENT}.csv", header=True, index=False, mode="w")


