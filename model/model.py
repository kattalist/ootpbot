import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential

sabr_train = pd.read_csv("model\data\processed.csv", names=["Name", "PA", "CON",
                                                 "GAP", "POW", "EYE", "K's",
                                                 "CON vL", "POW vL", "CON vR",
                                                 "POW vR", "BUN", "BFH", "SPE", "STE",
                                                 "WRC"])
sabr_train.sample(frac=1).reset_index(drop=True)
print(sabr_train)
sabr_train.drop('Name', axis=1, inplace=True)
sabr_train.drop('PA', axis=1, inplace=True)
sabr_train.drop('BUN', axis=1, inplace=True)
sabr_train.drop('BFH', axis=1, inplace=True)
sabr_train.drop('SPE', axis=1, inplace=True)
sabr_train.drop('STE', axis=1, inplace=True)
sabr_features = sabr_train.copy()
sabr_labels = sabr_features.pop('WRC')
sabr_features = np.array(sabr_features).astype('float32')
normalize = layers.Normalization()
normalize.adapt(sabr_features)

sabr_model = Sequential([normalize, layers.Dense(14), layers.Dense(8, activation='relu'), layers.Dense(1)])
sabr_model.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer = tf.keras.optimizers.Adam())
sabr_model.fit(sabr_features, sabr_labels, epochs=50, shuffle=True)

sabr_test = pd.read_csv("model\data\processed_2030.csv", names=["Name", "PA", "CON",
                                                 "GAP", "POW", "EYE", "K's",
                                                 "CON vL", "POW vL", "CON vR",
                                                 "POW vR", "BUN", "BFH", "SPE", "STE",
                                                 "WRC"])
sabr_test.drop('Name', axis=1, inplace=True)
sabr_test.drop('PA', axis=1, inplace=True)
sabr_test.drop('BUN', axis=1, inplace=True)
sabr_test.drop('BFH', axis=1, inplace=True)
sabr_test.drop('SPE', axis=1, inplace=True)
sabr_test.drop('STE', axis=1, inplace=True)
test_features = sabr_test.copy()
test_labels = test_features.pop('WRC')
test_features = np.array(test_features).astype('float32')
results = sabr_model.evaluate(test_features, test_labels, batch_size=2)
print("test loss, test acc:", results)

predictions = sabr_model.predict([50,50,50,50,50,50,50,50,50])
print(predictions)
sabr_model.save('model/default_model')