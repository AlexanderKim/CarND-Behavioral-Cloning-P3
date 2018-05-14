def load_data(dir):
    import csv
    import os
    import cv2
    import numpy as np
    from pprint import pprint

    csv_path = os.path.join(dir, 'driving_log.csv')
    img_path = os.path.join(dir, 'IMG')

    with open(csv_path) as csvfile:
        images_steering = [
            [ line[i].split('/')[-1], float(line[3]) ]
            for line in csv.reader(csvfile)
            for i in range(3)
            if line[0] != 'center'
        ]

    X_train = [
        cv2.imread(os.path.join(img_path, line[0]))
        for line in images_steering
    ]

    y_train = [line[1] for line in images_steering]

    X_train_flipped = [cv2.flip(img, 1) for img in X_train]
    y_train_reversed = y_train * -1

    for X, y in zip(X_train_flipped, y_train_reversed):
        X_train.append(X)
        y_train.append(y)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train

def train_model(X_train, y_train, output_path):
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, MaxPooling2D

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

    model.save(output_path)

import argparse
if __name__ == '__main__':
    argumets_parser = argparse.ArgumentParser()
    argumets_parser.add_argument('--dir', type=str, default='.')
    args = argumets_parser.parse_args()

    X_train, y_train = load_data(args.dir)

    train_model(X_train, y_train, 'model.h5')