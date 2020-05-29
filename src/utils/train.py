import tensorflow as tf
import tensorflow.keras.backend as kb
import csv

class TrainUnet():
    """ Train U-net model and save the results of the validation loss and dice loss in a CSV
    """

    def __init__(self, model, model_path:str, save_name:str, csv_session_path:str, X_train, Y_train, x_val, y_val,
                 batch_size:int, epochs:int):
        """Define the training of the model

        Parameters
        ----------
        model : model file keras
        Model file in keras which contains the predefined model to be fitted

        model_path : str
        Path to where the model should be saved. This happens every time the model improves on the validation accuracy.

        save_name : str
        The name of the model which is saved.

        csv_session_path : str
        The path to where the CSV session should be saved to

        X_train : tensor
        A tesnor containing all the training pictures. In this case normalized grey-scale images.

        Y_train : tensor
        A tensor containing the normalized ground truth pictures.

        x_val : tensor
        A tensor containing the validation images.

        y_val : str
        A tensor containing the ground truth images.

        batch_size : int
        The batch size used for training the algorithm.

        epochs : int
        A integer relating to how many epoch the algorithm will be trained on.
        """
        self.model = model
        self.model_path = model_path
        self.save_name = save_name
        self.csv_session_path = csv_session_path
        self.X_train = X_train
        self.Y_train = Y_train
        self.x_val = x_val
        self.y_val = y_val
        self.batch_size = batch_size
        self.epochs = epochs

    def save_train_ses(self, results):
        with open(self.csv_session_path + self.save_name + "-" + str(len(results.history["val_loss"])) + self.save_name +
                  ".csv", 'w', newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["loss", "dice_coef", "val_loss", "val_dice_coef"])
            for epoch in range(0, len(results.history["loss"])):
                writer.writerow([results.history["loss"][epoch], results.history['dice_coef'][epoch],
                                 results.history['val_loss'][epoch], results.history['val_dice_coef'][epoch]])
        return print("saved history as train-" + str(len(results.history["loss"])) + self.save_name + ".csv")


    def fit_unet(self):
        # ever want to save more models without oversaving add: -{epoch:02d}-{val_loss:.2f} to save name
        checkpointer = tf.keras.callbacks.ModelCheckpoint(self.model_path+self.save_name+"-{epoch:02d}-{val_loss:.2f}.h5", verbose=1,
                                                          save_best_only=True)
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=35, monitor='loss'), checkpointer]
        results = self.model.fit(self.X_train, self.Y_train, validation_data=(self.x_val, self.y_val),
                                 batch_size=self.batch_size, epochs=self.epochs, callbacks=callbacks)
        return results
