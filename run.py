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
URM = reader.load_urm2() 
ICM_channel = reader.load_icm_channel()
ICM_genre = reader.load_icm_genre()
ICM_subgenre = reader.load_icm_subgenre()
targets = reader.load_target()


####################### ISTANTIATE AND FIT THE HYBRID #######################

recommender = Massive(URM, ICM_channel,ICM_subgenre)
recommender.fit()  

################################ PRODUCE CSV ################################

f = open("recommendation.csv", "w+")
f.write("user_id,item_list\n")
for t in tqdm(targets):
    recommended_items = recommender.recommend(t, cutoff=10, remove_seen_flag=True)
    well_formatted = " ".join([str(x) for x in recommended_items])
    f.write(f"{t}, {well_formatted}\n")
