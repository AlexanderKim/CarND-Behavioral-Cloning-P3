import numpy as np
import cv2
import os
import sklearn


def load_data(dir):
    import csv

    csv_path = os.path.join(dir, 'driving_log.csv')
    # img_path = os.path.join(dir, 'IMG')

    with open(csv_path) as csvfile:
        images_steering = [
            [ line[i].split('/')[-1], float(line[3]) *1.5 ]
            for line in csv.reader(csvfile)
            for i in range(3)
            if line[0] != 'center'
        ]

    return images_steering

    # X_train = [
    #     cv2.imread(os.path.join(img_path, line[0]))
    #     for line in images_steering
    # ]
    #
    # y_train = [line[1] for line in images_steering]
    #
    # X_train_flipped = [cv2.flip(img, 1) for img in X_train]
    # y_train_reversed = y_train * -1
    #
    # for X, y in zip(X_train_flipped, y_train_reversed):
    #     X_train.append(X)
    #     y_train.append(y)
    #
    # X_train = np.array(X_train)
    # y_train = np.array(y_train)
    #
    # return X_train, y_train

def generator(samples, dir, batch_size=32):
    img_path = os.path.join(dir, 'IMG')

    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = os.path.join(img_path, batch_sample[0])
                image = cv2.imread(name)
                angle = float(batch_sample[1])

                if abs(angle) < 1.0:
                    continue

                images.append(image)
                angles.append(angle)

                images.append(cv2.flip(image, 1))
                angles.append(angle * -1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def compile_model():
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, MaxPooling2D

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))

    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model


import argparse
if __name__ == '__main__':
    argumets_parser = argparse.ArgumentParser()
    argumets_parser.add_argument('--dir', type=str, default='.')
    args = argumets_parser.parse_args()

    model = compile_model()

    # X_train, y_train = load_data(args.dir)
    # model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(load_data(args.dir), test_size=0.2)

    train_generator = generator(train_samples, args.dir, batch_size=32)
    validation_generator = generator(validation_samples, args.dir, batch_size=32)

    model.fit_generator(
        train_generator,
        samples_per_epoch=len(train_samples),
        validation_data = validation_generator,
        nb_val_samples = len(validation_samples), nb_epoch = 3
    )

    model.save('model.h5')