# Import libraries
import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf

# Import specific libraries
from keras import layers




def main(): 

    # Example 1: A first look at neural networks

    ### Listing 2.1: Loading the MNIST dataset in Keras
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()


    ### Listing 2.2: The network architecture
    model = tf.keras.Sequential([
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])


    ### Listing 2.3: The compilation step
    model.compile(optimizer="rmsprop",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    

    ### Listing 2.4: Preparing the image data
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype("float32") / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype("float32") / 255


    ### Listing 2.5: "Fitting" the model
    model.fit(train_images, train_labels, epochs=5, batch_size=128)


    ### Listing 2.6: Using the model to make predictions
    test_digits = test_images[0:10]
    predictions = model.predict(test_digits)
    print()
    print(predictions[0])
    print()
    print(predictions[0].argmax())
    print()
    print(test_labels[0])
    print()


    ### Listing 2.7: Evaluating the model on new data
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print()
    print(f"test_acc: {test_acc}")
    print()



if __name__ == "__main__":
    main()

    