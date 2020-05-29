import os
import os.path
import tensorflow as tf
import tensorflow.keras.backend as kb
import numpy as np
import cv2
import csv
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import math


class PredictUnet():
    """ Predict on pictures U-net model. The input images have to be of the original size (1024, 851) (you could build
    this in the functionality)
    """

    def __init__(self, model_path: str, probability: float, image_predict_path: str, upload_mask_path: str,
                 val_data_path: str, px_to_mm:float, save_result_path:str, save_result_name:str):
        """Define the u-net model and its parameters.

        Parameters
        ----------
        model_path : str
        Contains the path to the model saved in the models folder

        probability : float
        Contains a value for the probability of spark erosion in the image.

        image_predict_path: str
        The path to which image you would like to do predictions on

        upload_mask_path : str
        The path to where you would like to output example predictions

        val_data_path : str
        The path to where the validation or test set is for validating the algoritm

        ipx_to_mm: str
        Is the factor of ipx to mm from the picture

        save_result_path : str
        Path contains where the csv containing the IOU scores will be saved

        save_result_name : str
        The name of the CSV that will contain the evaluation scores
        """
        self.model_path = model_path
        self.probability = probability
        self.image_predict_path = image_predict_path
        self.upload_mask_path = upload_mask_path
        self.val_data_path = val_data_path
        self.px_to_mm = px_to_mm
        self.save_result_path = save_result_path
        self.save_result_name = save_result_name
        self.px2 = (px_to_mm*px_to_mm)

    def upload_mask_prediction_crossval(self):
        n = 0
        prediction_model = tf.keras.models.load_model(self.model_path, compile=False)
        for filename in tqdm(os.listdir(self.val_data_path)):
            if filename.endswith(".jpg") == True and filename.endswith("_predicted_mask.jpg") != True:
                img = cv2.imread(self.val_data_path + filename, cv2.IMREAD_GRAYSCALE)
                re_img = img[0:832, 224:800]
                # Now the prediction on the image, first shape data
                n += 1
                picturevalue = np.array(re_img)
                picturevalue = picturevalue.astype('float32')
                picturevalue /= 255.0
                picturevalue = picturevalue.reshape(-1, 832, 576, 1)

                # Predict the picture
                prediction = prediction_model.predict(picturevalue, verbose=1)

                # Hier doe je de probability, voor nu 50%
                binary_image = (prediction > self.probability).astype(np.uint8)

                # Reshape data
                binary_image = binary_image.reshape(832, 576)
                mask = binary_image * 255.0
                image = filename.replace(".jpg", "_predicted_mask.jpg")
                cv2.imwrite(self.upload_mask_path + image, mask)
        return

    def check_prediction_Crossval(self):
        for filename in tqdm(os.listdir(self.upload_mask_path)):
            if filename.endswith(".jpg") == True and filename.endswith("_predicted_mask.jpg") == True:
                im = cv2.imread(self.upload_mask_path + filename)

                # cv2.imshow(filename, im)
                # # Wait until a key is pressed:
                # cv2.waitKey(0)
                # # Destroy all created windows:
                # cv2.destroyAllWindows()

                imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                ret, thresh_image = cv2.threshold(imgray, 200, 255, 0)


                image = filename.replace("_predicted_mask.jpg", ".jpg")
                img = cv2.imread(self.val_data_path + image)
                if img.shape == (832, 576):
                    continue
                elif img.shape != (832, 576):
                    img = img[0:832, 224:800]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img = cv2.drawContours(img, contours, -1, (255, 255, 0), 2)


                # calculate area of pixels spark
                place_text = [(0, 20), (0, 40), (0, 60), (0, 80), (0, 100), (0, 120), (0, 140), (0, 160)]
                totale_area = 0
                height = 0
                tot_sparktale = 0
                count = 0
                for i in range(0, len(contours)):
                    count += 1
                    # this piece of code is to separate the spark areas in a two seperate binary image
                    black = np.zeros((832, 576, 3))
                    spark = cv2.drawContours(black, [contours[i]], -1, (255, 255, 255), 3)
                    spark = spark.astype('float32')
                    spark = cv2.cvtColor(spark, cv2.COLOR_BGR2GRAY)  # this is the specific spark region
                    # extract the area
                    area = round((cv2.contourArea(contours[i]))*self.px2, 2)
                    totale_area += area

                    # extract length and with from square around the mask in pixels x dpi
                    x, y, w, h = cv2.boundingRect(contours[i])
                    height_mm = round((h * self.px_to_mm), 2)
                    length_mm = round((w * self.px_to_mm), 2)
                    height += round(height_mm, 2)
                    height_plastic_plate = 6
                    sparktale = round((area * height_mm)/height_plastic_plate, 2)
                    tot_sparktale += sparktale

                    cv2.putText(img, "Area: "+str(area)+"  Height: "+str(height_mm)+"  Length: "+str(length_mm)+
                                "  Sparktale: "+str(sparktale), place_text[i], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                    # to draw the squire
                    # x, y, w, h = cv2.boundingRect(contours[i])
                    # spark = cv2.rectangle(spark, (x, y), (x + w, y + h), (255, 255, 0), 1)

                cv2.putText(img, "Total Sparktale " + str(round(tot_sparktale, 2)), place_text[count],
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                jojo = image
                jojo = jojo.replace("mask.jpg", "sparktale.jpg")
                cv2.imwrite(self.upload_mask_path + jojo, img)
                cv2.imshow(filename, img)
                # Wait until a key is pressed:
                cv2.waitKey(3000)
                # Destroy all created windows:
                cv2.destroyAllWindows()
        return print("Finished checking images!")

    def Validation_metrics(self):
        with open(self.save_result_path + self.save_result_name + ".csv", 'w', newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Picture", "Number", "Tp", "Tn", "Fp", "Fn", "accuracy", "Recall", "Precision",
                             "Specificity", "F1", "MCC", "IOU", "Dice"])
            prediction_model = tf.keras.models.load_model(self.model_path, compile=False)
            average_IOU = 0
            i = 0
            for image in os.listdir(self.val_data_path):
                if image.endswith("_color_mask.png"):
                    image = image.replace("_color_mask.png", ".jpg")
                    name = image
                    # nu voorspelling doen met model
                    y_hat = cv2.imread(self.val_data_path + image, cv2.IMREAD_GRAYSCALE)
                    if y_hat.shape == (832, 576):
                        continue
                    elif y_hat.shape != (832, 576):
                        y_hat = y_hat[0:832, 224:800]
                    picturevalue = np.array(y_hat)
                    picturevalue = picturevalue.astype('float32')
                    picturevalue /= 255.0
                    picturevalue = picturevalue.reshape(-1, 832, 576, 1)

                    # Predict the picture
                    y_pred = prediction_model.predict(picturevalue, verbose=1)
                    # Hier doe je de probability, voor nu 50%
                    y_pred = (y_pred > self.probability).astype(np.uint8)
                    y_pred = y_pred.reshape(832, 576)
                    y_pred = (y_pred * 255.0).astype('float32')

                    image = image.replace(".jpg", "_color_mask.png")
                    y_true = cv2.imread(self.val_data_path+image)

                    read = y_true[0:832, 224:800]  # this one is for unet.
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
                    read = cv2.cvtColor(read, cv2.COLOR_BGR2GRAY)
                    y_true = read.astype('float32')
                    y_true = ((y_true/250.0)*255.0)

                    # this all the measurements
                    y_true = y_true.round()
                    y_pred = y_pred.round() # y_pred.round()
                    pred = y_pred.ravel().tolist()
                    true = y_true.ravel().tolist()
                    cm = confusion_matrix(true, pred, labels=[255., 0.])
                    tp = int(cm[0][0])
                    tn = int(cm[1][1])
                    fp = int(cm[0][1])
                    fn = int(cm[1][0])
                    f1_score = (2 * tp) / ((2 * tp) + fp + fn)
                    recall = tp / (tp+fn) if (tp+fn) else 0    # sensitivity = recall
                    precision = tp / (tp+fp) if (tp+fp) else 0
                    specificity = tn / (tn+fp) if (tn+fp) else 0  # this is here if the model is very bad and prvents division by 0
                    acc = (tp + tn)/ (tp+tn+fp+fn) if (tp+tn+fp+fn) else 0
                    MCC = ((tp*tn)-(fp*fn))/(math.sqrt(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))) if (math.sqrt(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))) else 0

                    # Calculating by using image operations (not the best way)
                    intersection = cv2.bitwise_and(y_pred, y_true)
                    union = cv2.bitwise_or(y_pred, y_true)
                    IOU = ((np.sum(intersection)/np.sum(union))*100)
                    Dice = (2*np.sum(intersection)/ (np.sum(y_true)+np.sum(y_pred)))
                    i += 1
                    average_IOU = average_IOU+IOU

                    writer.writerow([name, i, tp, tn, fp, fn, acc, recall, precision, specificity, f1_score, MCC, IOU, Dice])

                    # # uncomment if you want to show images and its evaluation
                    # cv2.imshow("True_label", y_true)
                    # cv2.imshow("predicted_label", y_pred)
                    # cv2.imshow("Union", union)
                    # cv2.imshow("intersection", intersection)
                    # # Wait until a key is pressed:
                    # cv2.waitKey(0)
                    # # Destroy all created windows:
                    # cv2.destroyAllWindows()
            print("number of validation pictures= ", i)
            print("IOU score = ", round((average_IOU/i), 2))
        return round((average_IOU/i), 2)


