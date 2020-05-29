from src.utils.Predict_unet import PredictUnet
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "models/")
DATA_PATH_TRAINING_SESSION = os.path.join(ROOT_DIR, "data/Training_session/")
TESTSET_PATH = os.path.join(ROOT_DIR, "data/Testset/")
RESULTS_PATH = os.path.join(ROOT_DIR, "data/Results/")




# This is the algorithm you want to display and generate output for !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
model_name2 = "0_U-net_dice_seed3_feat6_batch2-26-0.11.h5"






predict = PredictUnet(model_path=MODEL_PATH + model_name2,
                      probability=0.5,
                      image_predict_path=TESTSET_PATH,
                      upload_mask_path=RESULTS_PATH,
                      val_data_path=TESTSET_PATH,
                      px_to_mm=0.264583333,
                      save_result_path=DATA_PATH_TRAINING_SESSION ,
                      save_result_name="testresults-")


predict.upload_mask_prediction_crossval()

predict.check_prediction_Crossval()
