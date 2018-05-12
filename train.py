def load_data(dir):
    import csv
    import os
    import cv2
    import numpy as np

    csv_path = os.path.join(dir, 'driving_log.csv')
    img_path = os.path.join(dir, 'IMG')

    with open(csv_path) as csvfile:
        images_steering = [
            [
                line[0].split('/')[-1],
                line[3]
            ]
            for line in csv.reader(csvfile)
        ]

    X_train = [
        cv2.imread(os.path.join(img_path, line[0]))
        for line in images_steering
    ]

    y_train = [line[1] for line in images_steering]

    return np.array(X_train), np.array(y_train)

def train_model(X_train, y_train, output_path):
    from keras.models import Sequential
    from keras.layers import Flatten, Dense

    model = Sequential()
    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

    model.save(output_path)

import argparse
if __name__ == '__main__':
    argumets_parser = argparse.ArgumentParser()
    argumets_parser.add_argument('--dir', type=str, default='.')
    args = argumets_parser.parse_args()

    X_train, y_train = load_data(args.dir)

    from pprint import pprint
    train_model(X_train, y_train, 'model.h5')