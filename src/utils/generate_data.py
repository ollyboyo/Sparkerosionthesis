import os
import os.path
import numpy as np
import cv2
from tqdm import tqdm
import random
from random import seed
from sklearn.model_selection import KFold


class ReadData():
    """ Read the data for training a model
    """

    def __init__(self, image_path, mask_path, validation_path, validation_split, set_seed, KFold_n:int):
        """Defines all the paths and validation split of the data. It will return a normalized dataset ready to be used
        for training an algorithm

        Parameters
        ----------
        image_path : str
        Path to the images

        mask_path : str
        path to the masks

        validation_path : str
        Path where validation pictures will be uploaded

        validation_split : float
        Is the split of the data, how much data will be used for validation.

        set_seed : int
        the seed of the training data.

        KFold_n : int
        The number of KFold cross validation splits you want to use in training the algorithm.
        """
        self.image_path = image_path
        self.mask_path = mask_path
        self.validation_path = validation_path
        self.validation_split = validation_split
        self.set_seed = set_seed
        self.KFold_n = KFold_n


    def create_data_pixelannotationtool_KFold_split(self):
        annotated = dict()
        seed(self.set_seed)
        i = 0
        for filename in os.listdir(self.mask_path):
            if filename.endswith("_color_mask.png"):
                image = filename.replace("_color_mask.png", ".jpg")
                image = image.replace(self.mask_path, "")
                annotated[i] = image, os.path.join(self.mask_path, filename)
                i += 1
        amount_of_pics = len(annotated)

        # this is for creating the kfold cross validation split and dictionary of picture name with mask ---------------
        kf = KFold(n_splits=5, random_state=self.set_seed, shuffle=True)
        split = np.zeros(amount_of_pics)
        i = 0
        split_indices = dict()
        for train, test in kf.split(split):
            split_indices[i] = train, test
            i += 1
        train_pictures = dict()
        val_pictures = dict()
        train, val = split_indices[self.KFold_n]
        for datapoint in train:
            train_pictures[annotated[datapoint][0]] = annotated[datapoint][1]
        for datapoint in val:
            val_pictures[annotated[datapoint][0]] = annotated[datapoint][1]
        # end you have create two dict with train and val pictures  ----------------------------------------------------
        print("")
        print("Save the validation data-set in sub folder k: "+str(self.KFold_n))
        # Save the validation images -----------------------------------------------------------------------------------
        if not os.path.exists(self.validation_path + str(self.KFold_n) + "/"):
            os.makedirs(self.validation_path + str(self.KFold_n) + "/")
        for path, value in tqdm(val_pictures.items()):
            read = cv2.imread(self.image_path + path, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(self.validation_path+str(self.KFold_n)+"/" + path, read)

            path = path.replace(".jpg", "_color_mask.png")
            read = cv2.imread(self.mask_path+path)
            cv2.imwrite(self.validation_path+str(self.KFold_n)+"/" + path, read)
        # Save the validation images -----------------------------------------------------------------------------------
        print("")
        print("Creating validation data-set")
        for path, value in tqdm(val_pictures.items()):
            # Dit eerste deel is voor pixelannotation mask
            read = cv2.imread(value)
            read = read[0:832, 224:800]  # this one is for unet.
            # Load the aerial image and convert to HSV colourspace
            hsv = cv2.cvtColor(read, cv2.COLOR_BGR2HSV)
            # Define lower and uppper limits of what we call "pink"
            pink_lo = np.array([0, 0, 0])
            pink_hi = np.array([141, 151, 231])
            # Mask image to only select pink
            mask = cv2.inRange(hsv, pink_lo, pink_hi)
            # Change image to black where we found pink
            read[mask > 0] = (0, 0, 0)
            # CONVERT to grayscale
            gray = cv2.cvtColor(read, cv2.COLOR_BGR2GRAY)
            # Convert to np Array
            annotetedvalue = np.array(gray)

            # Dit deel is voor de matching foto bij van de es las
            re_img = cv2.imread(self.image_path + path, cv2.IMREAD_GRAYSCALE)
            # Define the shape of the image.   original = re_img.shape
            re_img = re_img[0:832, 224:800]
            picturevalue = np.array(re_img)
            # Make dictionary with the values. (could save it)
            val_pictures[path] = annotetedvalue, picturevalue
        print("")
        print("Now normalize the data")
        x_val = np.zeros((len(val_pictures), 832, 576, 1), dtype=np.float32)
        y_val = np.zeros((len(val_pictures), 832, 576, 1), dtype=np.float32)
        n = -1
        for path in tqdm(val_pictures.items()):
            n += 1
            # for all the X train values
            X = path[1][1].astype('float32')
            X /= 255.0
            X = np.expand_dims(X, axis=2)
            x_val[n] = X

            # for all the Y-train values /250 cause 250 max and create 1!
            Y = path[1][0].astype('float32')
            Y /= 250.0
            Y = np.expand_dims(Y, axis=2)
            y_val[n] = Y
        print("")
        print("Create the training data-set")
        for path, value in tqdm(train_pictures.items()):
            # Dit eerste deel is voor pixelannotation mask
            read = cv2.imread(value)
            read = read[0:832, 224:800]  # this one is for unet.
            # Load the aerial image and convert to HSV colourspace
            hsv = cv2.cvtColor(read, cv2.COLOR_BGR2HSV)
            # Define lower and uppper limits of what we call "pink"
            pink_lo = np.array([0, 0, 0])
            pink_hi = np.array([141, 151, 231])
            # Mask image to only select pink
            mask = cv2.inRange(hsv, pink_lo, pink_hi)
            # Change image to black where we found pink
            read[mask > 0] = (0, 0, 0)
            # CONVERT to grayscale
            gray = cv2.cvtColor(read, cv2.COLOR_BGR2GRAY)
            # Convert to np Array
            annotetedvalue = np.array(gray)

            # Dit deel is voor de matching foto bij van de es las
            re_img = cv2.imread(self.image_path + path, cv2.IMREAD_GRAYSCALE)
            # Define the shape of the image.   original = re_img.shape
            re_img = re_img[0:832, 224:800]
            picturevalue = np.array(re_img)
            # Make dictionary with the values. (could save it)
            train_pictures[path] = annotetedvalue, picturevalue
        print("")
        print("Now normalize the data")
        X_train = np.zeros((len(train_pictures), 832, 576, 1), dtype=np.float32)
        Y_train = np.zeros((len(train_pictures), 832, 576, 1), dtype=np.float32)
        n = -1
        for path in tqdm(train_pictures.items()):
            n += 1
            # for all the X train values
            X = path[1][1].astype('float32')
            X /= 255.0
            X = np.expand_dims(X, axis=2)
            X_train[n] = X

            # for all the Y-train values /250 cause 250 max and create 1!
            Y = path[1][0].astype('float32')
            Y /= 250.0
            Y = np.expand_dims(Y, axis=2)
            Y_train[n] = Y
        return X_train, Y_train, x_val, y_val


    def create_data_pixelannotationtool(self):
        annotated = dict()
        val_dict = dict()
        seed(self.set_seed)
        for filename in os.listdir(self.mask_path):
            if filename.endswith("_color_mask.png"):
                image = filename.replace("_color_mask.png", ".jpg")
                image = image.replace(self.mask_path, "")
                annotated[image] = os.path.join(self.mask_path, filename)
        amount_of_pics = len(annotated)
        # create validation data split by using annotated images
        val_amount = round(len(annotated)*self.validation_split)
        print(amount_of_pics, "annotated pictures found")
        print((val_amount), "used for validation")
        print((len(annotated)-val_amount), "used for training")
        print("Creating validation data-set")
        for picture in tqdm(range(0, val_amount)):
            idx = random.randint(0, len(annotated)-1)
            pic_list = list(annotated)
            a_key = pic_list[idx]
            del annotated[a_key]

            # create x_val and y_val
            read = cv2.imread(self.image_path+a_key, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(self.validation_path + a_key, read)
            read = read[0:832, 224:800]
            picturevalue = np.array(read).astype('float32')

            a_key=a_key.replace(".jpg", "_color_mask.png")
            read = cv2.imread(self.mask_path+a_key)
            cv2.imwrite(self.validation_path+a_key, read)
            read = read[0:832, 224:800]
            hsv = cv2.cvtColor(read, cv2.COLOR_BGR2HSV)
            pink_lo = np.array([0, 0, 0])
            pink_hi = np.array([141, 151, 231])
            mask = cv2.inRange(hsv, pink_lo, pink_hi)
            read[mask > 0] = (0, 0, 0)
            gray = cv2.cvtColor(read, cv2.COLOR_BGR2GRAY)
            annotatedvalue = np.array(gray).astype('float32')
            val_dict[a_key] = picturevalue, annotatedvalue
        print("")
        print("Now normalize the val-data")
        x_val = np.zeros((len(val_dict), 832, 576, 1), dtype=np.float32)
        y_val = np.zeros((len(val_dict), 832, 576, 1), dtype=np.float32)
        n = -1
        for path in tqdm(val_dict.items()):
            n += 1
            y = path[1][1].astype('float32')
            y /= 250.0
            y = np.expand_dims(y, axis=2)
            y_val[n] = y

            x = path[1][0].astype('float32')
            x /= 255.0
            x = np.expand_dims(x, axis=2)
            x_val[n] = x
        print("")
        print("Create the training data-set")
        for path, value in tqdm(annotated.items()):
            # Dit eerste deel is voor pixelannotation mask
            read = cv2.imread(value)
            read = read[0:832, 224:800] # this one is for unet.
            # Load the aerial image and convert to HSV colourspace
            hsv = cv2.cvtColor(read, cv2.COLOR_BGR2HSV)
            # Define lower and uppper limits of what we call "pink"
            pink_lo = np.array([0, 0, 0])
            pink_hi = np.array([141, 151, 231])
            # Mask image to only select pink
            mask = cv2.inRange(hsv, pink_lo, pink_hi)
            # Change image to black where we found pink
            read[mask > 0] = (0, 0, 0)
            # CONVERT to grayscale
            gray = cv2.cvtColor(read, cv2.COLOR_BGR2GRAY)
            # Convert to np Array
            annotetedvalue = np.array(gray)

            # Dit deel is voor de matching foto bij van de es las
            re_img = cv2.imread(self.image_path + path, cv2.IMREAD_GRAYSCALE)
            # Define the shape of the image.   original = re_img.shape
            re_img = re_img[0:832, 224:800]
            picturevalue = np.array(re_img)
            # Make dictionary with the values. (could save it)
            annotated[path] = annotetedvalue, picturevalue
        print("")
        print("Now normalize the data")
        X_train = np.zeros((len(annotated), 832, 576, 1), dtype=np.float32)
        Y_train = np.zeros((len(annotated), 832, 576, 1), dtype=np.float32)
        n = -1
        for path in tqdm(annotated.items()):
            n += 1
            # for all the X train values
            X = path[1][1].astype('float32')
            X /= 255.0
            X = np.expand_dims(X, axis=2)
            X_train[n] = X

            # for all the Y-train values /250 cause 250 max and create 1!
            Y = path[1][0].astype('float32')
            Y /= 250.0
            Y = np.expand_dims(Y, axis=2)
            Y_train[n] = Y
        return X_train, Y_train, x_val, y_val
