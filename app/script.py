# %% md
## In this tutorial, I will show how to code a license plate recognizer for  license plates using deep learning and some image processing.
### Find the detailed explanation of the project in this blog: https://towardsdatascience.com/ai-based-license-plate-detector-de9d48ca8951?source=friends_link&sk=a2cbd70e630f6dc3d030e3bae34d98ef
# %%
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import f1_score
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
# Upload an image
import urllib.request
import re
import pyrebase

# check if file exits
import pathlib

config = {
    "apiKey": "AIzaSyC1VK73kHLRM6Picu6YjGg6pYbTcFt9gEs",
    "authDomain": "learn-plus-fyp.firebaseapp.com",
    "projectId": "learn-plus-fyp",
    "databaseURL": "",
    "storageBucket": "learn-plus-fyp.appspot.com",
    "messagingSenderId": "497930965512",
    "appId": "1:497930965512:web:f949e47f58b4e7113af7bd",
    "measurementId": "G-TD2JKTFK9N",
}
firebase = pyrebase.initialize_app(config)
storage = firebase.storage()


def number_plate_recognition(image_url, epochs=2):
        image_url= str(image_url)
    # Image Received
    # try:
        pattern = "https://firebasestorage.googleapis.com/v0/b/learn-plus-fyp.appspot.com/o/python%2f(.*?)\?"
        print(image_url)
        substring = re.search(pattern, image_url).group(1)
        image_type = substring[-3:]
        image_id = substring[:-4]

        # Read image url
        urllib.request.urlretrieve(image_url, "temp." + image_type)

        # %%
        # Loads the data required for detecting the license plates from cascade classifier.
        plate_cascade = cv2.CascadeClassifier("license_plate.xml")
        if plate_cascade.empty():
            print("empty")
        else:
            print("ok")
        print(plate_cascade)

        # add the path to 'india_license_plate.xml' file.
        # %%
        def detect_plate(img, text=''):  # the function detects and perfors blurring on the number plate.
            plate_img = img.copy()
            roi = img.copy()
            plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.2,
                                                        minNeighbors=7)  # detects numberplates and returns the coordinates and dimensions of detected license plate's contours.
            for (x, y, w, h) in plate_rect:
                roi_ = roi[y:y + h, x:x + w, :]  # extracting the Region of Interest of license plate for blurring.
                plate = roi[y:y + h, x:x + w, :]
                cv2.rectangle(plate_img, (x + 2, y), (x + w - 3, y + h - 5), (51, 181, 155),
                              3)  # finally representing the detected contours by drawing rectangles around the edges.
            if text != '':
                plate_img = cv2.putText(plate_img, text, (x - w // 2, y - h // 2),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (51, 181, 155), 1, cv2.LINE_AA)

            return plate_img, plate  # returning the processed image.

        # %%
        # Testing the above function
        def display(img_, title=''):
            img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
            fig = plt.figure(figsize=(10, 6))
            ax = plt.subplot(111)
            ax.imshow(img)
            plt.axis('off')
            plt.title(title)
            plt.show()

        img = cv2.imread('temp.' + image_type)
        display(img, 'input image')
        # %%
        # Getting plate prom the processed image
        output_img, plate = detect_plate(img)
        # %%
        display(output_img, 'detected license plate in the input image')

        plate_contour = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        plt.imshow(plate_contour)
        plt.axis('off')
        plt.savefig('plate_contour.jpg')

        # %%
        display(plate, 'extracted license plate from the image')

        # %%
        # Match contours to license plate or character template
        def find_contours(dimensions, img):
            # Find all contours in the image
            cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Retrieve potential dimensions
            lower_width = dimensions[0]
            upper_width = dimensions[1]
            lower_height = dimensions[2]
            upper_height = dimensions[3]

            # Check largest 5 or  15 contours for license plate or character respectively
            cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

            ii = cv2.imread('contour.jpg')

            x_cntr_list = []
            target_contours = []
            img_res = []
            for cntr in cntrs:
                # detects contour in binary image and returns the coordinates of rectangle enclosing it
                intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

                # checking the dimensions of the contour to filter out the characters by contour's size
                if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height:
                    x_cntr_list.append(
                        intX)  # stores the x coordinate of the character's contour, to used later for indexing the contours

                    char_copy = np.zeros((44, 24))
                    # extracting each character using the enclosing rectangle's coordinates.
                    char = img[intY:intY + intHeight, intX:intX + intWidth]
                    char = cv2.resize(char, (20, 40))

                    cv2.rectangle(ii, (intX, intY), (intWidth + intX, intY + intHeight), (50, 21, 200), 2)
                    plt.imshow(ii, cmap='gray')

                    # Make result formatted for classification: invert colors
                    char = cv2.subtract(255, char)

                    # Resize the image to 24x44 with black border
                    char_copy[2:42, 2:22] = char
                    char_copy[0:2, :] = 0
                    char_copy[:, 0:2] = 0
                    char_copy[42:44, :] = 0
                    char_copy[:, 22:24] = 0

                    img_res.append(char_copy)  # List that stores the character's binary image (unsorted)

            # Return characters on ascending order with respect to the x-coordinate (most-left character first)
            plt.axis('off')
            plt.savefig('plate_segmented.jpg')
            plt.show()

            # arbitrary function that stores sorted list of character indeces
            indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
            img_res_copy = []
            for idx in indices:
                img_res_copy.append(img_res[idx])  # stores character images according to their index
            img_res = np.array(img_res_copy)

            return img_res

        # %%
        # Find characters in the resulting images
        def segment_characters(image):
            # Preprocess cropped license plate image
            img_lp = cv2.resize(image, (333, 75))
            img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
            _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
            img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))

            LP_WIDTH = img_binary_lp.shape[0]
            LP_HEIGHT = img_binary_lp.shape[1]

            # Make borders white
            img_binary_lp[0:3, :] = 255
            img_binary_lp[:, 0:3] = 255
            img_binary_lp[72:75, :] = 255
            img_binary_lp[:, 330:333] = 255

            # Estimations of character contours sizes of cropped license plates
            dimensions = [LP_WIDTH / 6,
                          LP_WIDTH / 2,
                          LP_HEIGHT / 10,
                          2 * LP_HEIGHT / 3]
            plt.imshow(img_binary_lp, cmap='gray')
            plt.show()
            cv2.imwrite('contour.jpg', img_binary_lp)

            # Get contours within cropped license plate
            char_list = find_contours(dimensions, img_binary_lp)

            return char_list

        # %%
        # Let's see the segmented characters
        char = segment_characters(plate)

        # %%
        for i in range(len(char)):
            plt.subplot(1, 10, i + 1)
            plt.imshow(char[i], cmap='gray')
            plt.axis('off')
        # %% md
        ### Model for characters
        # %%
        import keras.backend as K

        train_datagen = ImageDataGenerator(rescale=1. / 255, width_shift_range=0.1, height_shift_range=0.1)
        path = 'data'
        train_generator = train_datagen.flow_from_directory(
            path + '/train',  # this is the target directory
            target_size=(28, 28),  # all images will be resized to 28x28
            batch_size=1,
            class_mode='sparse')

        validation_generator = train_datagen.flow_from_directory(
            path + '/val',  # this is the target directory
            target_size=(28, 28),  # all images will be resized to 28x28 batch_size=1,
            class_mode='sparse')

        # %%
        # Metrics for checking the model performance while training
        def f1score(y, y_pred):
            return f1_score(y, tf.math.argmax(y_pred, axis=1), average='micro')

        def custom_f1score(y, y_pred):
            return tf.py_function(f1score, (y, y_pred), tf.double)

        # %%
        K.clear_session()
        model = Sequential()
        model.add(Conv2D(16, (22, 22), input_shape=(28, 28, 3), activation='relu', padding='same'))
        model.add(Conv2D(32, (16, 16), input_shape=(28, 28, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (8, 8), input_shape=(28, 28, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (4, 4), input_shape=(28, 28, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(36, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.Adam(lr=0.0001),
                      metrics=[custom_f1score])
        # %%
        model.summary()

        # %%
        class stop_training_callback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                if (logs.get('val_custom_f1score') > 0.99):
                    self.model.stop_training = True

        # %%
        batch_size = 1
        callbacks = [stop_training_callback()]
        model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_data=validation_generator,
            epochs=epochs, verbose=1, callbacks=callbacks)

        # %%
        # Predicting the output
        def fix_dimension(img):
            new_img = np.zeros((28, 28, 3))
            for i in range(3):
                new_img[:, :, i] = img
            return new_img

        def show_results():
            dic = {}
            characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            for i, c in enumerate(characters):
                dic[i] = c

            output = []
            for i, ch in enumerate(char):  # iterating over the characters
                img_ = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
                img = fix_dimension(img_)
                img = img.reshape(1, 28, 28, 3)  # preparing image for the model
                predict_y = model.predict(img)[0]  # predicting the class
                y_ = np.argmax(predict_y, axis=0)
                character = dic[y_]  #
                output.append(character)  # storing the result in a list

            plate_number = ''.join(output)

            return plate_number

        results = show_results()
        # %%
        # Segmented characters and their predicted value.
        plt.figure(figsize=(10, 6))
        for i, ch in enumerate(char):
            img = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
            plt.subplot(3, 4, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(f'predicted: {show_results()[i]}')
            plt.axis('off')
        plt.savefig('predictions.png')
        plt.show()

        # %%
        # plate_number = show_results()
        # output_img, plate = detect_plate(img, plate_number)
        # display(output_img, 'detected license plate number in the input image')
        # %%

        def upload_firebase(output):
            path_on_cloud = "python/" + image_id + '-' + output
            url = storage.child(path_on_cloud).put(output)
            return 'https://firebasestorage.googleapis.com/v0/b/learn-plus-fyp.appspot.com/o/python%2F' + image_id + '-' + output + '?alt=media&token=' + \
                   url['downloadTokens']

        if pathlib.Path('plate_contour.jpg').exists() and pathlib.Path('predictions.png').exists() and pathlib.Path(
                'plate_segmented.jpg').exists():
            print("File exist")
            plate_contour_url = upload_firebase('plate_contour.jpg')
            predictions_url = upload_firebase('predictions.png')
            plate_segmented_url = upload_firebase('plate_segmented.jpg')
        else:
            print("File not exist")

        return {
            "plate_contour": plate_contour_url,
            "plate_segmented": plate_segmented_url,
            "predictions": predictions_url,
            "plate_number": results
        }
    # except Exception as e:
    #     print({"error": e, "message": "Server Down"})
    #     return {"error": e, "message": "Server Down"}

image_url='https://firebasestorage.googleapis.com/v0/b/learn-plus-fyp.appspot.com/o/python%2fcar8.jpg?alt=media&token=0af22fe9-358e-44bf-a296-86074d16e734'
print(number_plate_recognition(image_url,0))