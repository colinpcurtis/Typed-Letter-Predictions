from tensorflow import keras
import pandas as pd
import numpy as np

data = pd.read_csv("letter-recognition.data", index_col=False)

labels = data.iloc[:, 0].values.tolist()  # gives us the letters we want to predict on 

values = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10, "K": 11, "L": 12,
          "M": 13, "N": 14, "O": 15, "P": 16, "Q": 17, "R": 18, "S": 19, "T": 20, "U": 21, "V": 22, "W": 23,
          "X": 24, "Y": 25, "Z": 26}  # we need the data in our model to be numeric so we can do analysis on it 


def get_key(dictionary, value):
    for key in dictionary.keys():
        if dictionary[key] == value:
            return key


def to_numeric(letters):  # switches letters in dataset to numbers
    new_letters = []
    global values
    for letter in letters:
        new_letters.append(values[letter])
    return new_letters


labels = to_numeric(labels)

attributes = data.iloc[:, 1:].values.tolist()

x_train = attributes[2000:]  # 18000 train, 2000 test
x_test = attributes[:2000]

y_train = labels[2000:]
y_test = labels[:2000]

model = keras.Sequential([
    keras.layers.Dense(16, activation="selu"),
    keras.layers.Dense(16, activation="selu"),
    keras.layers.Dense(16, activation="selu"),
    keras.layers.Dense(27, activation="softmax")
])

# In a perfect world we would have more inputs neurons than output neurons,
# but the way our data is set up doesn't make this possible.  We compensate for this by having more hidden layers
# This is not optimal, but it allows us to do what we need to do

model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# below loop allows us to run the model many times and save whatever one gives us the best accuracy
# stops us from having to retrain the model each time we want to predict on new data
# we could delete this, but it might be helpful to keep in case we feel like retraining the model using different activations or layer setups

"""best = 0  
for n in range(100):
    print(n)
    model.fit(x_train, y_train, batch_size=6, epochs=20, verbose=0)
    accuracy = model.evaluate(x_test, y_test) # model.evaluate returns list of [loss, metrics]
    print(accuracy[1])
    if accuracy[1] > best:
        model.save("best_model")
    else:
        continue"""


model = keras.models.load_model("best_model")  # loads the best model we saved

prediction = model.predict(x_test)


def display_predictions(num):  # prints out the actual letter and prediction, and tells us what we predicted incorrectly.
    wrong = []
    for i in range(num):
        print(i, ", actual:", get_key(values, y_test[i]), ", prediction:", get_key(values, np.argmax(prediction[i])))
        if get_key(values, y_test[i]) != get_key(values, np.argmax(prediction[i])):
            wrong.append(i)
    print(wrong, len(wrong))  # allows us to see what predictions we got wrong and how many there were


display_predictions(2000)  # lets us see the predictions on our test dataset
