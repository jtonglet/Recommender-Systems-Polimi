################################# IMPORT RECOMMENDERS #################################

from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.PureSVDRecommender import PureSVDRecommender
from Recommenders.RP3betaRecommender import RP3betaRecommender
from Recommenders.P3alphaRecommender import P3alphaRecommender
from Recommenders.UserKNNCFRecommender import UserKNNCFRecommender

################################## IMPORT LIBRARIES ##################################

from numpy import linalg as LA
import scipy.sparse as sps
from tqdm import tqdm
import numpy as np


#################################### HYBRID CLASS ####################################
#Keep this as the config for the best current model

class Massive(BaseRecommender):

    RECOMMENDER_NAME = "Massive"

    def __init__(self, URM_train, ICM_channel,ICM_subgenre, EASE_R_Model, IALS_Model, SLIM_Model): #Expects a pretrained EASE_R model
        super(Massive, self).__init__(URM_train)
        self.ICM_channel = ICM_channel
        self.ICM_subgenre = ICM_subgenre
        self.EASE_R = EASE_R_Model
        self.IALS = IALS_Model
        self.SLIM = SLIM_Model
        
        self.URM_aug = sps.vstack([self.URM_train, ICM_channel.T, ICM_subgenre.T])
        
        

    def fit(self,  
            lambda1 = 10, lambda2 = 1, lambda3 = 1, lambda4 = 1, lambda5 = 1, lambda6 = 0, lambda7 =1, topK_CBF1 = 7000, shrink1 = 10000, similarity1 = 'cosine', 
            feature_weighting1 = 'tfidf'):
      
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4   #SLIM was so good that it could be augmented a little bit 
        self.lambda5 = lambda5
        self.lambda6 = lambda6
        self.lambda7 = lambda7

        # Stack and normalize URM and ICM
        #ICM = similaripy.normalization.bm25plus(self.ICM.copy())
        
        # Instantiate the recommenders     
        self.ItemCBF = ItemKNNCBFRecommender(self.URM_train, self.ICM_channel)
        self.SVD = PureSVDRecommender(self.URM_aug)
        self.RP3 = RP3betaRecommender(self.URM_aug)
        self.P3 = P3alphaRecommender(self.URM_aug)
        self.CF = UserKNNCFRecommender(self.URM_aug)
        

 
        
    
                                              
                                              
        self.ItemCBF.fit(topK = topK_CBF1, shrink =  shrink1, similarity = similarity1, feature_weighting = feature_weighting1)
        self.CF.fit(topK = 432, shrink =34, similarity = 'jaccard')
        self.SVD.fit(num_factors = 27)
        self.RP3.fit(alpha = 0.775, beta = 0.495 , implicit = True, topK = 111)
        self.P3.fit(alpha = 0.625, topK = 66, implicit = True)
       
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
                   
            w4 = self.SVD._compute_item_score(user_id_array[i], items_to_compute) 
            w4 /= LA.norm(w4,2) 

            w5 = self.SLIM._compute_item_score(user_id_array[i], items_to_compute) 
            w5 /= LA.norm(w5,2) 

            w6 = self.RP3._compute_item_score(user_id_array[i], items_to_compute) 
            w6 /= LA.norm(w6,2) 

            w7 = self.P3._compute_item_score(user_id_array[i], items_to_compute) 
            w7 /= LA.norm(w7,2)

            w8 = self.CF._compute_item_score(user_id_array[i], items_to_compute) 
            w8 /= LA.norm(w8,2)

            w = (self.lambda1 * w12 + self.lambda2 *w3 + self.lambda3 * w4 +
            self.lambda4 * w5+self.lambda5 * w6+self.lambda6 * w7+self.lambda7 * w8 )
                
            
            item_weights[i,:] = w 
            
            
        return item_weights
    
    