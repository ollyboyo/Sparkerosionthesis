from src.utils.model_unet import ModelUnet
from src.utils.generate_data import ReadData
from src.utils.train import TrainUnet
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "models/")
DATA_PATH_TRAINING =  os.path.join(ROOT_DIR, "data/Trainingset/")
DATA_PATH_TEST =  os.path.join(ROOT_DIR, "data/Testset/")
DATA_PATH_TRAINING_SESSION =  os.path.join(ROOT_DIR, "data/Training_session/")


#------------------------- setup loop for training and evaluating all the cross val at once ----------------------------
i = 0
model_name = "__U-net_dice_seed3_feat6_batch2"
model_path = MODEL_PATH
for training_round in range(0, 5):
    # ------------------------------define the model -------------------------------------------------------------------
    features = [6, 12, 24, 48, 96]
    unet = ModelUnet(loss_def="dice_loss", image_with=576, image_hight=832, amount_features=features, label_n=1)



    model = unet.U_net_model() # THIS IS WHERE YOU CAN CHANGE TO OTHER MODELS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    model.summary()






    # ------------------------------define the model -------------------------------------------------------------------

    # ------------------------------read in the data -------------------------------------------------------------------
    readdata = ReadData(image_path=DATA_PATH_TRAINING,
                        mask_path=DATA_PATH_TRAINING,
                        validation_path=DATA_PATH_TRAINING_SESSION,
                        validation_split=0.0, set_seed=3, KFold_n=i)
    X_train, Y_train, x_val, y_val = readdata.create_data_pixelannotationtool_KFold_split()
    # ------------------------------read in the data -------------------------------------------------------------------

    # ------------------------------train the model  -------------------------------------------------------------------
    model_name = str(i) + model_name[1:]
    train = TrainUnet(model=model, model_path=model_path, save_name=model_name,
                      csv_session_path=DATA_PATH_TRAINING_SESSION+str(i)+"/",
                      X_train=X_train, Y_train=Y_train, x_val=x_val, y_val=y_val, batch_size=2, epochs=2)
    history = train.fit_unet()
    # ------------------------------train the model  -------------------------------------------------------------------

    # ------------------------------save the model session -------------------------------------------------------------
    train.save_train_ses(history)
    # ------------------------------save the model session -------------------------------------------------------------
    i += 1



