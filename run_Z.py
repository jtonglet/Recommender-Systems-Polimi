################################## IMPORTS ##################################

from Utils.Evaluator import EvaluatorHoldout
from Utils.DataSplitter import DataSplitter
from Utils.DataReader import DataReader
from Massive_Hybrid import Massive
from Recommenders.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.IALSRecommender import IALSRecommender
from Recommenders.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from tqdm import tqdm
from time import time

################################# READ DATA #################################
reader = DataReader()
splitter = DataSplitter()
URM = reader.load_urm2() #Load the full URM
ICM_channel = reader.load_icm_channel()
ICM_genre = reader.load_icm_genre()
ICM_subgenre = reader.load_icm_subgenre()
targets = reader.load_target()


####################### ISTANTIATE AND FIT THE HYBRID #######################

#Load the pretrained models
EASE_R = EASE_R_Recommender(URM)
EASE_R.load_model("saved_models/EASE_R_full_ICM")

IALS = IALSRecommender(URM)
IALS.load_model("saved_models/IALS_full_ICM") #Train an IALS on the full dataset : TO DO 


SLIM = SLIM_BPR_Cython(URM)
SLIM.load_model("saved_models/slim_full_ICM")



recommender = Massive(URM, ICM_channel,ICM_subgenre, EASE_R, IALS, SLIM)
recommender.fit(        lambda1 = 10,
                        lambda2 = 1,  #IALS
                        lambda3 = 0,    # SVD
                        lambda4 = 1,  #SLIM
                        lambda5 = 1, #RP3
                        lambda6=0,  #P3 is bad
                        lambda7=1  #CF
                       
               )  #Friday submission : adjust weights

################################ PRODUCE CSV ################################

f = open("submissions/recommendation07_01_D.csv", "w+")
f.write("user_id,item_list\n")
for t in tqdm(targets):
    recommended_items = recommender.recommend(t, cutoff=10, remove_seen_flag=True)
    well_formatted = " ".join([str(x) for x in recommended_items])
    f.write(f"{t}, {well_formatted}\n")