from src.utils.Predict_unet import PredictUnet
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "models/")
DATA_PATH_TRAINING_SESSION = os.path.join(ROOT_DIR, "data/Training_session/")
TESTSET_PATH = os.path.join(ROOT_DIR, "data/Testset/")


# THIS HAS TO BE CHANGED TO THE MODELS THAT NEED TO BE EVALUATED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
model_name0 = "0_U-net_dice_seed3_feat6_batch2-26-0.11.h5"
model_name1 = "0_U-net_dice_seed3_feat6_batch2-26-0.11.h5"
model_name2 = "0_U-net_dice_seed3_feat6_batch2-26-0.11.h5"
model_name3 = "0_U-net_dice_seed3_feat6_batch2-26-0.11.h5"
model_name4 = "0_U-net_dice_seed3_feat6_batch2-26-0.11.h5"

models = [model_name0, model_name1, model_name2, model_name3, model_name4]


i = 0
for number in range(0, 5):
    # ------------------------------save the evaluation metrics --------------------------------------------------------
    print("Evaluating model:", models[i])
    predict = PredictUnet(model_path=MODEL_PATH + models[i],
                          probability=0.5,
                          image_predict_path=TESTSET_PATH,
                          upload_mask_path=TESTSET_PATH,
                          val_data_path=TESTSET_PATH,
                          px_to_mm=0.264583333,
                          save_result_path=DATA_PATH_TRAINING_SESSION + str(i) + "/",
                          save_result_name=str(i)+"testresults-"+ str(models[i]))
    predict.Validation_metrics()
    i += 1


