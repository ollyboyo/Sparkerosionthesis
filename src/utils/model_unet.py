import tensorflow as tf
from src.utils.loss_functions import dice_coef, dice_coef_loss

class ModelUnet():
    """ Define all the U-net model architectures.
    """

    def __init__(self, loss_def:str, image_with:int, image_hight:int, amount_features:list, label_n:int):
        """Define the u-net model and its parameters.

        Parameters
        ----------
        loss_def : str
        Defines which loss function to use with the model.
        You have two options: "dice_loss" or ""binary_crossentropy""

        image_with : int
        This contains the with of your image in an int variable

        image_hight : int
        This contains the hight of your image in an int variable

        amount_features : list of 5 int
        Contians a list of 5 integers which represent the features convolutional block.

        label_n: int
        This variable contains the amount of classes that need to be predicted within the image
        """
        self.loss_def = loss_def
        self.image_with = image_with
        self.image_hight = image_hight
        self.amount_features = amount_features
        self.label_n = label_n

    def baseline_CNN(self):
        # encoder ------------------------------------------------------------------------------------------------------
        # !!!!!!!!! Height and width of input images should be divisible by 32 for all models !!!!!!!!!  Do in test!!!!!
        inputs = tf.keras.layers.Input((self.image_hight, self.image_with, 1))  # (832, 576, 1)

        c1 = tf.keras.layers.Conv2D(self.amount_features[0], (3, 3), activation='relu', padding="same")(
            inputs)  # [20, 40, 80, 160, 320]
        c1 = tf.keras.layers.Conv2D(self.amount_features[0], (3, 3), activation='relu', padding="same")(c1)
        p1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c1)

        c2 = tf.keras.layers.Conv2D(self.amount_features[1], (3, 3), activation='relu', padding="same")(p1)
        c2 = tf.keras.layers.Conv2D(self.amount_features[1], (3, 3), activation='relu', padding="same")(c2)
        p2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c2)

        c3 = tf.keras.layers.Conv2D(self.amount_features[2], (3, 3), activation='relu', padding="same")(p2)
        c3 = tf.keras.layers.Conv2D(self.amount_features[2], (3, 3), activation='relu', padding="same")(c3)
        p3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c3)

        c4 = tf.keras.layers.Conv2D(self.amount_features[3], (3, 3), activation='relu', padding="same")(p3)
        c4 = tf.keras.layers.Conv2D(self.amount_features[3], (3, 3), activation='relu', padding="same")(c4)
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = tf.keras.layers.Conv2D(self.amount_features[4], (3, 3), activation='relu', padding="same")(p4)
        c5 = tf.keras.layers.Conv2D(self.amount_features[4], (3, 3), activation='relu', padding="same")(c5)

        # Decoder --------------------------------------------------------------------------------------------------------------
        u6 = tf.keras.layers.Conv2DTranspose(self.amount_features[3], (2, 2), strides=(2, 2), padding="same")(c5)
        # no concatination
        c6 = tf.keras.layers.Conv2D(self.amount_features[3], (3, 3), activation='relu', padding="same")(u6)
        c6 = tf.keras.layers.Conv2D(self.amount_features[3], (3, 3), activation='relu', padding="same")(c6)

        u7 = tf.keras.layers.Conv2DTranspose(self.amount_features[2], (2, 2), strides=(2, 2), padding="same")(c6)
        # no concatination
        c7 = tf.keras.layers.Conv2D(self.amount_features[2], (3, 3), activation='relu', padding="same")(u7)
        c7 = tf.keras.layers.Conv2D(self.amount_features[2], (3, 3), activation='relu', padding="same")(c7)

        u8 = tf.keras.layers.Conv2DTranspose(self.amount_features[1], (2, 2), strides=(2, 2), padding="same")(c7)
        # no concatination
        c8 = tf.keras.layers.Conv2D(self.amount_features[1], (3, 3), activation='relu', padding="same")(u8)
        c8 = tf.keras.layers.Conv2D(self.amount_features[1], (3, 3), activation='relu', padding="same")(c8)

        u9 = tf.keras.layers.Conv2DTranspose(self.amount_features[0], (2, 2), strides=(2, 2), padding="same")(c8)
        # no concatination
        c9 = tf.keras.layers.Conv2D(self.amount_features[0], (3, 3), activation='relu', padding="same")(u9)
        c9 = tf.keras.layers.Conv2D(self.amount_features[0], (3, 3), activation='relu', padding="same")(c9)

        outputs = tf.keras.layers.Conv2D(self.label_n, (1, 1), activation="sigmoid")(c9)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        if self.loss_def == "dice_loss":
            model.compile(optimizer="adam", loss=dice_coef_loss, metrics=[dice_coef])
        elif self.loss_def == "binary_crossentropy":
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        return model

    def U_net_model(self, pretrained_weights=None):
        # encoder ------------------------------------------------------------------------------------------------------
        # !!!!!!!!! Height and width of input images should be divisible by 32 for all models !!!!!!!!!  Do in test!!!!!
        inputs = tf.keras.layers.Input((self.image_hight, self.image_with, 1))  # (832, 576, 1)

        c1 = tf.keras.layers.Conv2D(self.amount_features[0], (3, 3), activation='relu', padding="same")(inputs)
        c1 = tf.keras.layers.Conv2D(self.amount_features[0], (3, 3), activation='relu', padding="same")(c1)
        p1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c1)

        c2 = tf.keras.layers.Conv2D(self.amount_features[1], (3, 3), activation='relu', padding="same")(p1)
        c2 = tf.keras.layers.Conv2D(self.amount_features[1], (3, 3), activation='relu', padding="same")(c2)
        p2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c2)

        c3 = tf.keras.layers.Conv2D(self.amount_features[2], (3, 3), activation='relu', padding="same")(p2)
        c3 = tf.keras.layers.Conv2D(self.amount_features[2], (3, 3), activation='relu', padding="same")(c3)
        p3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c3)

        c4 = tf.keras.layers.Conv2D(self.amount_features[3], (3, 3), activation='relu', padding="same")(p3)
        c4 = tf.keras.layers.Conv2D(self.amount_features[3], (3, 3), activation='relu', padding="same")(c4)
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = tf.keras.layers.Conv2D(self.amount_features[4], (3, 3), activation='relu', padding="same")(p4)
        c5 = tf.keras.layers.Conv2D(self.amount_features[4], (3, 3), activation='relu', padding="same")(c5)

        # Decoder --------------------------------------------------------------------------------------------------------------
        u6 = tf.keras.layers.Conv2DTranspose(self.amount_features[3], (2, 2), strides=(2, 2), padding="same")(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        c6 = tf.keras.layers.Conv2D(self.amount_features[3], (3, 3), activation='relu', padding="same")(u6)
        c6 = tf.keras.layers.Conv2D(self.amount_features[3], (3, 3), activation='relu', padding="same")(c6)

        u7 = tf.keras.layers.Conv2DTranspose(self.amount_features[2], (2, 2), strides=(2, 2), padding="same")(c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        c7 = tf.keras.layers.Conv2D(self.amount_features[2], (3, 3), activation='relu', padding="same")(u7)
        c7 = tf.keras.layers.Conv2D(self.amount_features[2], (3, 3), activation='relu', padding="same")(c7)

        u8 = tf.keras.layers.Conv2DTranspose(self.amount_features[1], (2, 2), strides=(2, 2), padding="same")(c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        c8 = tf.keras.layers.Conv2D(self.amount_features[1], (3, 3), activation='relu', padding="same")(u8)
        c8 = tf.keras.layers.Conv2D(self.amount_features[1], (3, 3), activation='relu', padding="same")(c8)

        u9 = tf.keras.layers.Conv2DTranspose(self.amount_features[0], (2, 2), strides=(2, 2), padding="same")(c8)
        u9 = tf.keras.layers.concatenate([u9, c1])
        c9 = tf.keras.layers.Conv2D(self.amount_features[0], (3, 3), activation='relu', padding="same")(u9)
        c9 = tf.keras.layers.Conv2D(self.amount_features[0], (3, 3), activation='relu', padding="same")(c9)

        outputs = tf.keras.layers.Conv2D(self.label_n, (1, 1), activation="sigmoid")(c9)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        if self.loss_def == "dice_loss":
            model.compile(optimizer="adam", loss=dice_coef_loss, metrics=[dice_coef])
        elif self.loss_def == "binary_crossentropy":
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        if (pretrained_weights):
            model.load_weights(pretrained_weights)

        return model

    def Resu_net_model(self):
        # encoder ------------------------------------------------------------------------------------------------------
        inputs = tf.keras.layers.Input((self.image_hight, self.image_with, 1))  # (832, 576, 1)

        # block 1 ------------------------------------------------------------------------------------------------------
        one = tf.keras.layers.Conv2D(self.amount_features[0], [1, 1], padding="same")(inputs)

        c1 = tf.keras.layers.Conv2D(self.amount_features[0], (3, 3), activation='relu', padding="same")(one)
        c1 = tf.keras.layers.Conv2D(self.amount_features[0], (3, 3), activation='relu', padding="same")(c1)

        adition1 = tf.keras.layers.add([c1, one])
        p1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(adition1)

        # block 2 ------------------------------------------------------------------------------------------------------
        two = tf.keras.layers.Conv2D(self.amount_features[1], [1, 1], padding="same")(p1)

        c2 = tf.keras.layers.Conv2D(self.amount_features[1], (3, 3), activation='relu', padding="same")(two)
        c2 = tf.keras.layers.Conv2D(self.amount_features[1], (3, 3), activation='relu', padding="same")(c2)

        adition2 = tf.keras.layers.add([c2, two])
        p2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(adition2)

        # block 3 ------------------------------------------------------------------------------------------------------
        three = tf.keras.layers.Conv2D(self.amount_features[2], [1, 1], padding="same")(p2)

        c3 = tf.keras.layers.Conv2D(self.amount_features[2], (3, 3), activation='relu', padding="same")(three)
        c3 = tf.keras.layers.Conv2D(self.amount_features[2], (3, 3), activation='relu', padding="same")(c3)

        adition3 = tf.keras.layers.add([c3, three])
        p3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(adition3)

        # block 4 ------------------------------------------------------------------------------------------------------
        four = tf.keras.layers.Conv2D(self.amount_features[3], [1, 1], padding="same")(p3)

        c4 = tf.keras.layers.Conv2D(self.amount_features[3], (3, 3), activation='relu', padding="same")(four)
        c4 = tf.keras.layers.Conv2D(self.amount_features[3], (3, 3), activation='relu', padding="same")(c4)

        adition4 = tf.keras.layers.add([c4, four])
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(adition4)

        # block 5 ------------------------------------------------------------------------------------------------------
        five = tf.keras.layers.Conv2D(self.amount_features[4], [1, 1], padding="same")(p4)
        c5 = tf.keras.layers.Conv2D(self.amount_features[4], (3, 3), activation='relu', padding="same")(five)
        c5 = tf.keras.layers.Conv2D(self.amount_features[4], (3, 3), activation='relu', padding="same")(c5)

        adition5 = tf.keras.layers.add([c5, five])
        # here no maxpooling p because it is the middele layer! dus straks in loop if equal as len(features)

        # decoder ------------------------------------------------------------------------------------------------------
        # block 6 ------------------------------------------------------------------------------------------------------
        u1 = tf.keras.layers.Conv2DTranspose(self.amount_features[3], (2, 2), strides=(2, 2), padding="same")(adition5)

        u2 = tf.keras.layers.concatenate([u1, adition4])
        six = tf.keras.layers.Conv2D(self.amount_features[3], [1, 1], padding="same")(u2)
        c6 = tf.keras.layers.Conv2D(self.amount_features[3], (3, 3), activation='relu', padding="same")(six)
        c6 = tf.keras.layers.Conv2D(self.amount_features[3], (3, 3), activation='relu', padding="same")(c6)

        p6 = tf.keras.layers.add([c6, six])

        # block 7 ------------------------------------------------------------------------------------------------------
        u3 = tf.keras.layers.Conv2DTranspose(self.amount_features[2], (2, 2), strides=(2, 2), padding="same")(p6)

        u4 = tf.keras.layers.concatenate([u3, adition3])
        seven = tf.keras.layers.Conv2D(self.amount_features[2], [1, 1], padding="same")(u4)
        c7 = tf.keras.layers.Conv2D(self.amount_features[2], (3, 3), activation='relu', padding="same")(seven)
        c7 = tf.keras.layers.Conv2D(self.amount_features[2], (3, 3), activation='relu', padding="same")(c7)

        p7 = tf.keras.layers.add([c7, seven])

        # block 8 ------------------------------------------------------------------------------------------------------
        u5 = tf.keras.layers.Conv2DTranspose(self.amount_features[1], (2, 2), strides=(2, 2), padding="same")(p7)

        u6 = tf.keras.layers.concatenate([u5, adition2])
        eight = tf.keras.layers.Conv2D(self.amount_features[1], [1, 1], padding="same")(u6)
        c8 = tf.keras.layers.Conv2D(self.amount_features[1], (3, 3), activation='relu', padding="same")(eight)
        c8 = tf.keras.layers.Conv2D(self.amount_features[1], (3, 3), activation='relu', padding="same")(c8)

        p8 = tf.keras.layers.add([c8, eight])

        # block 9 ------------------------------------------------------------------------------------------------------
        u7 = tf.keras.layers.Conv2DTranspose(self.amount_features[1], (2, 2), strides=(2, 2), padding="same")(p8)
        u8 = tf.keras.layers.concatenate([u7, adition1])

        nine = tf.keras.layers.Conv2D(self.amount_features[0], [1, 1], padding="same")(u8)
        c9 = tf.keras.layers.Conv2D(self.amount_features[0], (3, 3), activation='relu', padding="same")(nine)
        c9 = tf.keras.layers.Conv2D(self.amount_features[0], (3, 3), activation='relu', padding="same")(c9)
        p9 = tf.keras.layers.add([c9, nine])

        c10 = tf.keras.layers.Conv2D(self.label_n, (1, 1), activation="sigmoid", padding='same')(p9)
        model = tf.keras.models.Model(inputs=inputs, outputs=c10)

        if self.loss_def == "dice_loss":
            model.compile(optimizer="adam", loss=dice_coef_loss, metrics=[dice_coef])
        elif self.loss_def == "binary_crossentropy":
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        return model

    def RU_net_model(self):
        # encoder ------------------------------------------------------------------------------------------------------
        inputs = tf.keras.layers.Input((self.image_hight, self.image_with, 1))  # (832, 576, 1)

        # block 1 ------------------------------------------------------------------------------------------------------
        one = tf.keras.layers.Conv2D(self.amount_features[0], [1, 1], padding="same")(inputs)
        # Unfolded_Recurrent_Convolutional_layer -----------------------------------------------------------------------
        c1 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(one)
        adition = tf.keras.layers.add([c1, one])
        c2 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition)
        adition2 = tf.keras.layers.add([c2, one])
        c3 = tf.keras.layers.Conv2D(self.amount_features[0], (3, 3), activation='relu', padding="same")(adition2)
        # Unfolded_Recurrent_Convolutional_layer2 ----------------------------------------------------------------------
        c4 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(c3)
        adition3 = tf.keras.layers.add([c4, adition2])
        c5 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition3)
        adition4 = tf.keras.layers.add([c5, adition2])
        c6 = tf.keras.layers.Conv2D(self.amount_features[0], (3, 3), activation='relu', padding="same")(adition4)
        # Unfolded_Recurrent_Convolutional_layer2 ----------------------------------------------------------------------
        p1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c6)

        # block 2 ------------------------------------------------------------------------------------------------------
        two = tf.keras.layers.Conv2D(self.amount_features[1], [1, 1], padding="same")(p1)
        c7 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(two)
        adition6 = tf.keras.layers.add([c7, two])
        c8 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition6)
        adition7 = tf.keras.layers.add([c8, two])
        c9 = tf.keras.layers.Conv2D(self.amount_features[1], (3, 3), activation='relu', padding="same")(adition7)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        c10 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(c9)
        adition8 = tf.keras.layers.add([c10, adition7])
        c11 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition8)
        adition9 = tf.keras.layers.add([c11, adition7])
        c12 = tf.keras.layers.Conv2D(self.amount_features[1], (3, 3), activation='relu', padding="same")(adition9)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        p2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c12)

        # block 3 ------------------------------------------------------------------------------------------------------
        three = tf.keras.layers.Conv2D(self.amount_features[2], [1, 1], padding="same")(p2)
        c13 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(three)
        adition11 = tf.keras.layers.add([c13, three])
        c14 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition11)
        adition12 = tf.keras.layers.add([c14, three])
        c15 = tf.keras.layers.Conv2D(self.amount_features[2], (3, 3), activation='relu', padding="same")(adition12)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        c16 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(c15)
        adition13 = tf.keras.layers.add([c16, adition12])
        c17 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition13)
        adition14 = tf.keras.layers.add([c17, adition12])
        c18 = tf.keras.layers.Conv2D(self.amount_features[2], (3, 3), activation='relu', padding="same")(adition14)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        p3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c18)

        # block 4 ------------------------------------------------------------------------------------------------------
        four = tf.keras.layers.Conv2D(self.amount_features[3], [1, 1], padding="same")(p3)
        c19 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(four)
        adition16 = tf.keras.layers.add([c19, four])
        c20 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition16)
        adition17 = tf.keras.layers.add([c20, four])
        c21 = tf.keras.layers.Conv2D(self.amount_features[3], (3, 3), activation='relu', padding="same")(adition17)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        c22 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(c21)
        adition18 = tf.keras.layers.add([c22, adition17])
        c23 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition18)
        adition19 = tf.keras.layers.add([c23, adition17])
        c24 = tf.keras.layers.Conv2D(self.amount_features[3], (3, 3), activation='relu', padding="same")(adition19)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c24)

        # block 5 ------------------------------------------------------------------------------------------------------
        five = tf.keras.layers.Conv2D(self.amount_features[4], [1, 1], padding="same")(p4)
        c25 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(five)
        adition21 = tf.keras.layers.add([c25, five])
        c26 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition21)
        adition22 = tf.keras.layers.add([c26, five])
        c27 = tf.keras.layers.Conv2D(self.amount_features[4], (3, 3), activation='relu', padding="same")(adition22)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        c28 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(c27)
        adition23 = tf.keras.layers.add([c28, adition22])
        c29 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition23)
        adition24 = tf.keras.layers.add([c29, adition22])
        c30 = tf.keras.layers.Conv2D(self.amount_features[4], (3, 3), activation='relu', padding="same")(adition24)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------

        # decoder ------------------------------------------------------------------------------------------------------
        u1 = tf.keras.layers.Conv2DTranspose(self.amount_features[3], (2, 2), strides=(2, 2), padding="same")(c30)
        u2 = tf.keras.layers.concatenate([u1, c24])

        # block 6 ------------------------------------------------------------------------------------------------------
        six = tf.keras.layers.Conv2D(self.amount_features[3], [1, 1], padding="same")(u2)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        c31 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(six)
        adition26 = tf.keras.layers.add([c31, six])
        c32 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition26)
        adition27 = tf.keras.layers.add([c32, six])
        c33 = tf.keras.layers.Conv2D(self.amount_features[3], (3, 3), activation='relu', padding="same")(adition27)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        c34 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(c33)
        adition28 = tf.keras.layers.add([c34, adition27])
        c35 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition28)
        adition29 = tf.keras.layers.add([c35, adition27])
        c36 = tf.keras.layers.Conv2D(self.amount_features[3], (3, 3), activation='relu', padding="same")(adition29)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------

        u3 = tf.keras.layers.Conv2DTranspose(self.amount_features[2], (2, 2), strides=(2, 2), padding="same")(c36)
        u4 = tf.keras.layers.concatenate([u3, c18])

        # block 7 ------------------------------------------------------------------------------------------------------
        seven = tf.keras.layers.Conv2D(self.amount_features[2], [1, 1], padding="same")(u4)
        c37 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(seven)
        adition31 = tf.keras.layers.add([c37, seven])
        c38 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition31)
        adition32 = tf.keras.layers.add([c38, seven])
        c39 = tf.keras.layers.Conv2D(self.amount_features[2], (3, 3), activation='relu', padding="same")(adition32)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        c40 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(c39)
        adition33 = tf.keras.layers.add([c40, adition32])
        c41 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition33)
        adition34 = tf.keras.layers.add([c41, adition32])
        c42 = tf.keras.layers.Conv2D(self.amount_features[2], (3, 3), activation='relu', padding="same")(adition34)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------

        u5 = tf.keras.layers.Conv2DTranspose(self.amount_features[1], (2, 2), strides=(2, 2), padding="same")(c42)
        u6 = tf.keras.layers.concatenate([u5, c12])

        # block 8 ------------------------------------------------------------------------------------------------------
        eight = tf.keras.layers.Conv2D(self.amount_features[1], [1, 1], padding="same")(u6)
        c43 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(eight)
        adition36 = tf.keras.layers.add([c43, eight])
        c44 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition36)
        adition37 = tf.keras.layers.add([c44, eight])
        c45 = tf.keras.layers.Conv2D(self.amount_features[1], (3, 3), activation='relu', padding="same")(adition37)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        c46 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(c45)
        adition38 = tf.keras.layers.add([c46, adition37])
        c47 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition38)
        adition39 = tf.keras.layers.add([c47, adition37])
        c48 = tf.keras.layers.Conv2D(self.amount_features[1], (3, 3), activation='relu', padding="same")(adition39)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------

        u7 = tf.keras.layers.Conv2DTranspose(self.amount_features[1], (2, 2), strides=(2, 2), padding="same")(c48)
        u8 = tf.keras.layers.concatenate([u7, c6])

        # block 9 ------------------------------------------------------------------------------------------------------
        nine = tf.keras.layers.Conv2D(self.amount_features[0], [1, 1], padding="same")(u8)
        c49 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(nine)
        adition41 = tf.keras.layers.add([c49, nine])
        c50 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition41)
        adition42 = tf.keras.layers.add([c50, nine])
        c51 = tf.keras.layers.Conv2D(self.amount_features[0], (3, 3), activation='relu', padding="same")(adition42)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        c52 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(c51)
        adition43 = tf.keras.layers.add([c52, adition42])
        c53 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition43)
        adition44 = tf.keras.layers.add([c53, adition42])
        c54 = tf.keras.layers.Conv2D(self.amount_features[0], (3, 3), activation='relu', padding="same")(adition44)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------

        c55 = tf.keras.layers.Conv2D(self.label_n, (1, 1), activation="sigmoid", padding='same')(c54)
        model = tf.keras.models.Model(inputs=inputs, outputs=c55)

        if self.loss_def == "dice_loss":
            model.compile(optimizer="adam", loss=dice_coef_loss, metrics=[dice_coef])
        elif self.loss_def == "binary_crossentropy":
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        return model

    def R2U_net_model(self):
        # encoder ------------------------------------------------------------------------------------------------------
        inputs = tf.keras.layers.Input((self.image_hight, self.image_with, 1))  # (832, 576, 1)

        # block 1 ------------------------------------------------------------------------------------------------------
        one = tf.keras.layers.Conv2D(self.amount_features[0], [1, 1], padding="same")(inputs)
        c1 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(one)
        adition = tf.keras.layers.add([c1, one])
        c2 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition)
        adition2 = tf.keras.layers.add([c2, one])
        c3 = tf.keras.layers.Conv2D(self.amount_features[0], (3, 3), activation='relu', padding="same")(adition2)   # deze klopt ook niet denk ik
        # Unfolded_Recurrent_Convolutional_layer2 ----------------------------------------------------------------------
        c4 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(c3)
        adition3 = tf.keras.layers.add([c4, adition2])
        c5 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition3)
        adition4 = tf.keras.layers.add([c5, adition2])
        c6 = tf.keras.layers.Conv2D(self.amount_features[0], (3, 3), activation='relu', padding="same")(adition4)
        # Unfolded_Recurrent_Convolutional_layer2 ----------------------------------------------------------------------
        adition5 = tf.keras.layers.add([c6, one])
        p1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(adition5)

        # block 2 ------------------------------------------------------------------------------------------------------
        two = tf.keras.layers.Conv2D(self.amount_features[1], [1, 1], padding="same")(p1)

        c7 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(two)
        adition6 = tf.keras.layers.add([c7, two])
        c8 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition6)
        adition7 = tf.keras.layers.add([c8, two])
        c9 = tf.keras.layers.Conv2D(self.amount_features[1], (3, 3), activation='relu', padding="same")(adition7)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        c10 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(c9)
        adition8 = tf.keras.layers.add([c10, adition7])
        c11 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition8)
        adition9 = tf.keras.layers.add([c11, adition7])
        c12 = tf.keras.layers.Conv2D(self.amount_features[1], (3, 3), activation='relu', padding="same")(adition9)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        adition10 = tf.keras.layers.add([c12, two])
        p2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(adition10)

        # block 3 ------------------------------------------------------------------------------------------------------
        three = tf.keras.layers.Conv2D(self.amount_features[2], [1, 1], padding="same")(p2)

        c13 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(three)
        adition11 = tf.keras.layers.add([c13, three])
        c14 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition11)
        adition12 = tf.keras.layers.add([c14, three])
        c15 = tf.keras.layers.Conv2D(self.amount_features[2], (3, 3), activation='relu', padding="same")(adition12)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        c16 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(c15)
        adition13 = tf.keras.layers.add([c16, adition12])
        c17 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition13)
        adition14 = tf.keras.layers.add([c17, adition12])
        c18 = tf.keras.layers.Conv2D(self.amount_features[2], (3, 3), activation='relu', padding="same")(adition14)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        adition15 = tf.keras.layers.add([c18, three])
        p3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(adition15)

        # block 4 ------------------------------------------------------------------------------------------------------
        four = tf.keras.layers.Conv2D(self.amount_features[3], [1, 1], padding="same")(p3)
        c19 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(four)
        adition16 = tf.keras.layers.add([c19, four])
        c20 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition16)
        adition17 = tf.keras.layers.add([c20, four])
        c21 = tf.keras.layers.Conv2D(self.amount_features[3], (3, 3), activation='relu', padding="same")(adition17)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        c22 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(c21)
        adition18 = tf.keras.layers.add([c22, adition17])
        c23 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition18)
        adition19 = tf.keras.layers.add([c23, adition17])
        c24 = tf.keras.layers.Conv2D(self.amount_features[3], (3, 3), activation='relu', padding="same")(adition19)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        adition20 = tf.keras.layers.add([c24, four])
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(adition20)

        # block 5 ------------------------------------------------------------------------------------------------------
        five = tf.keras.layers.Conv2D(self.amount_features[4], [1, 1], padding="same")(p4)
        c25 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(five)
        adition21 = tf.keras.layers.add([c25, five])
        c26 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition21)
        adition22 = tf.keras.layers.add([c26, five])
        c27 = tf.keras.layers.Conv2D(self.amount_features[4], (3, 3), activation='relu', padding="same")(adition22)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        c28 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(c27)
        adition23 = tf.keras.layers.add([c28, adition22])
        c29 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition23)
        adition24 = tf.keras.layers.add([c29, adition22])
        c30 = tf.keras.layers.Conv2D(self.amount_features[4], (3, 3), activation='relu', padding="same")(adition24)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        p5 = tf.keras.layers.add([c30, five])



        # decoder ------------------------------------------------------------------------------------------------------
        u1 = tf.keras.layers.Conv2DTranspose(self.amount_features[3], (2, 2), strides=(2, 2), padding="same")(p5)
        # tf.keras.layers.UpSampling2D could also be used less computational expensive
        u2 = tf.keras.layers.concatenate([u1, adition20])

        # block 6 ------------------------------------------------------------------------------------------------------
        six = tf.keras.layers.Conv2D(self.amount_features[3], [1, 1], padding="same")(u2)
        c31 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(six)
        adition26 = tf.keras.layers.add([c31, six])
        c32 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition26)
        adition27 = tf.keras.layers.add([c32, six])
        c33 = tf.keras.layers.Conv2D(self.amount_features[3], (3, 3), activation='relu', padding="same")(adition27)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        c34 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(c33)
        adition28 = tf.keras.layers.add([c34, adition27])
        c35 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition28)
        adition29 = tf.keras.layers.add([c35, adition27])
        c36 = tf.keras.layers.Conv2D(self.amount_features[3], (3, 3), activation='relu', padding="same")(adition29)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        p6 = tf.keras.layers.add([c36, six])

        u3 = tf.keras.layers.Conv2DTranspose(self.amount_features[2], (2, 2), strides=(2, 2), padding="same")(p6)
        # tf.keras.layers.UpSampling2D could also be used less computational expensive
        u4 = tf.keras.layers.concatenate([u3, adition15])

        # block 7 ------------------------------------------------------------------------------------------------------
        seven = tf.keras.layers.Conv2D(self.amount_features[2], [1, 1], padding="same")(u4)
        c37 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(seven)
        adition31 = tf.keras.layers.add([c37, seven])
        c38 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition31)
        adition32 = tf.keras.layers.add([c38, seven])
        c39 = tf.keras.layers.Conv2D(self.amount_features[2], (3, 3), activation='relu', padding="same")(adition32)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        c40 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(c39)
        adition33 = tf.keras.layers.add([c40, adition32])
        c41 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition33)
        adition34 = tf.keras.layers.add([c41, adition32])
        c42 = tf.keras.layers.Conv2D(self.amount_features[2], (3, 3), activation='relu', padding="same")(adition34)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        p7 = tf.keras.layers.add([c42, seven])

        u5 = tf.keras.layers.Conv2DTranspose(self.amount_features[1], (2, 2), strides=(2, 2), padding="same")(
            p7)  # tf.keras.layers.UpSampling2D could also be used less computational expensive
        u6 = tf.keras.layers.concatenate([u5, adition10])

        # block 8 ------------------------------------------------------------------------------------------------------
        eight = tf.keras.layers.Conv2D(self.amount_features[1], [1, 1], padding="same")(u6)
        c43 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(eight)
        adition36 = tf.keras.layers.add([c43, eight])
        c44 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition36)
        adition37 = tf.keras.layers.add([c44, eight])
        c45 = tf.keras.layers.Conv2D(self.amount_features[1], (3, 3), activation='relu', padding="same")(adition37)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        c46 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(c45)
        adition38 = tf.keras.layers.add([c46, adition37])
        c47 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition38)
        adition39 = tf.keras.layers.add([c47, adition37])
        c48 = tf.keras.layers.Conv2D(self.amount_features[1], (3, 3), activation='relu', padding="same")(adition39)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        p8 = tf.keras.layers.add([c48, eight])

        u7 = tf.keras.layers.Conv2DTranspose(self.amount_features[1], (2, 2), strides=(2, 2), padding="same")(
            p8)  # tf.keras.layers.UpSampling2D could also be used less computational expensive
        u8 = tf.keras.layers.concatenate([u7, adition5])

        # block 9 ------------------------------------------------------------------------------------------------------
        nine = tf.keras.layers.Conv2D(self.amount_features[0], [1, 1], padding="same")(u8)
        c49 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(nine)
        adition41 = tf.keras.layers.add([c49, nine])
        c50 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition41)
        adition42 = tf.keras.layers.add([c50, nine])
        c51 = tf.keras.layers.Conv2D(self.amount_features[0], (3, 3), activation='relu', padding="same")(adition42)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        c52 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(c51)
        adition43 = tf.keras.layers.add([c52, adition42])
        c53 = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding="same")(adition43)
        adition44 = tf.keras.layers.add([c53, adition42])
        c54 = tf.keras.layers.Conv2D(self.amount_features[0], (3, 3), activation='relu', padding="same")(adition44)
        # Unfolded_Recurrent_Convolutional_layer  ----------------------------------------------------------------------
        p9 = tf.keras.layers.add([c54, nine])

        c55 = tf.keras.layers.Conv2D(self.label_n, (1, 1), activation="sigmoid", padding='same')(p9)
        model = tf.keras.models.Model(inputs=inputs, outputs=c55)


        if self.loss_def == "dice_loss":
            model.compile(optimizer="adam", loss=dice_coef_loss, metrics=[dice_coef])
        elif self.loss_def == "binary_crossentropy":
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        return model

