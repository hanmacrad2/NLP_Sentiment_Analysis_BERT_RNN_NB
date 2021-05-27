import numpy as np
import tensorflow as tf
import pandas as pd
import pickle 
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, f1_score, roc_curve, auc
import string

def pickle_result(result, filename):
    """ Pickle the result and save to a file """
    pickle.dump(result, open('roc_'+filename+".p", "wb"))


FILE_TRAIN = '../data/train_formatted.csv'
FILE_TEST = '../data/test_formatted.csv'
MAX_LEN = 50

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)

# Open and store the files
train = pd.read_csv(FILE_TRAIN,
    encoding='utf-8')
test = pd.read_csv(FILE_TEST,
    encoding='utf-8')

print(train.head())
features_train = train.pop("text").to_numpy()
labels_train = train.pop("polarity").to_numpy()

features_test = test.pop("text").to_numpy()
labels_test = test.pop("polarity").to_numpy()



try:
    # Try to load an already trained model
    model = tf.keras.models.load_model("model")
except:

    # If unsuccessful create and train a new one

    # Create the input vector with the encored
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=None,
        output_sequence_length=MAX_LEN)

    encoder.adapt(features_train)


    # Define and compile the model
    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=128,
            mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(1e-5),
                metrics=['accuracy'])

    # Train the model
    history = model.fit(features_train,labels_train,epochs=15,
                        validation_split=0.2,
                        validation_steps=30
                        )

    # Save the model to file
    model.save("model",save_format='tf')



# Evaluation

test_loss, test_acc = model.evaluate(features_test,labels_test)
model_predictions = model.predict_classes(features_test)

print("")
print("")
print("")
print("Model Scores")
print("Validation loss: {}".format(test_loss))
print("Validation Accuracy: {}".format(test_acc))
print("Validation F1-Score: {}".format(f1_score(model_predictions, labels_test,average='micro')))
print("Classification Report")
print(confusion_matrix(model_predictions, labels_test))
print(classification_report(model_predictions, labels_test))

print(labels_test)
print(model_predictions)

model.predict_proba(features_test)
fpr, tpr, thresholds = roc_curve(labels_test, model.predict_proba(features_test))
auc = auc(fpr, tpr)
print('AUC: %s' % auc)

# If you want to save the results to a file uncomment lines belows
pickle_result(fpr, 'fpr')
pickle_result(tpr, 'tpr')
pickle_result(thresholds, 'thresholds')
