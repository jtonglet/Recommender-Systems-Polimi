################################# IMPORT RECOMMENDERS #################################

from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.IALSRecommender import IALSRecommender
from Recommenders.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.RP3betaRecommender import RP3betaRecommender
from Recommenders.UserKNNCFRecommender import UserKNNCFRecommender


################################## IMPORT LIBRARIES ##################################

from numpy import linalg as LA
import scipy.sparse as sps
from tqdm import tqdm
import numpy as np

#################################### HYBRID CLASS ####################################

class Hybrid(BaseRecommender):

    RECOMMENDER_NAME = "Hybrid"

    def __init__(self, URM_train, ICM_channel,ICM_subgenre): 
        super(Massive, self).__init__(URM_train)
        self.ICM_channel = ICM_channel
        self.ICM_subgenre = ICM_subgenre
        
        self.URM_aug = sps.vstack([self.URM_train, ICM_channel.T, ICM_subgenre.T])
        
        

    def fit(self,  
            lambda1 = 10, 
            lambda2 = 1, 
            lambda3 = 1, 
            lambda4 = 1,
            lambda5 = 1, 
            ):
      
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4   
        self.lambda5 = lambda5


        
        # Instantiate the recommenders     
        self.ItemCBF = ItemKNNCBFRecommender(self.URM_train, self.ICM_channel)
        self.EASE_R = EASE_R_Recommender(self.URM_aug)
        self.IALS = IALSRecommender(self.URM_aug)
        self.SLIM = SLIM_BPR_Cython(self.URM_aug)
        self.RP3 = RP3betaRecommender(self.URM_aug)
        self.CF = UserKNNCFRecommender(self.URM_aug)
                                                    
                                              
        self.ItemCBF.fit(topK = 7000, 
                         shrink =  10000, 
                         similarity = 'jaccard', 
                         feature_weighting = 'tfidf')
        
        self.EASE_R.fit()
        
        self.IALS.fit(num_factors = 34)
        
        self.SLIM.fit(   )
        
        self.RP3.fit(alpha = 0.775,
                     beta = 0.495 , 
                     implicit = True, 
                     topK = 111)
        
        self.CF.fit(topK = 432, 
                    shrink =34,
                    similarity = 'jaccard')
        
       
    def _compute_item_score(self,
                            user_id_array, 
                            items_to_compute = None
                           ):

        item_weights = np.empty([len(user_id_array), 18059])
       
        for i in tqdm(range(len(user_id_array))):
            
            w1 = self.ItemCBF._compute_item_score(user_id_array[i], items_to_compute) 
            w1 /= LA.norm(w1,2)

            w2 = self.EASE_R._compute_item_score(user_id_array[i], items_to_compute)  
            w2 /= LA.norm(w2,2)
            
            w12 = 0.05 * w1 + 0.95 * w2
                        
            w3 = self.IALS._compute_item_score(user_id_array[i], items_to_compute) 
            w3 /= LA.norm(w3,2)
                             
            w4 = self.SLIM._compute_item_score(user_id_array[i], items_to_compute) 
            w4 /= LA.norm(w4,2) 

            w5 = self.RP3._compute_item_score(user_id_array[i], items_to_compute) 
            w5 /= LA.norm(w5,2) 

            w6 = self.CF._compute_item_score(user_id_array[i], items_to_compute) 
            w6 /= LA.norm(w6,2)

            w = self.lambda1 * w12 + self.lambda2 *w3 + self.lambda3 * w4 + self.lambda4 * w5 + self.lambda5 * w6
                        
            item_weights[i,:] = w 
            
            
        return item_weights
    
    
