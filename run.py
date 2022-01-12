################################## IMPORTS ##################################

from Utils.DataReader import DataReader
from Hybrid import Hybrid
from tqdm import tqdm
from time import time

################################# READ DATA #################################

reader = DataReader()
splitter = DataSplitter()
URM = reader.load_urm_sps() 
ICM_channel = reader.load_icm_channel()
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
