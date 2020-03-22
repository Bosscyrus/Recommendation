import numpy as np
from lightfm import LightFM
from lightfm.datasets import fetch_movielens
 
data = fetch_movielens(min_rating=4.0)
# min rating is the minimum ratin we'll want to include in our data i.e we're coleting the movies for 4 or heigher
#our fetch_movielens stores our data as string so lets print out the testing and training data
#print(repr(data['train']))
#print(repr(data['test']))

# using the loss function it measures the differnce btw or model prediction and the desired output
# we'll minimize it during training so our model gets more acurate over time
# creating model
model = LightFM(loss='warp') # a loss function warp = weighted approximate rank pairwise it looks at the rating of each useers 
# and predict ratings for each

# content based + collaborative system = HYBRID

# train model we'll use the fit method which takes 3 parameters the number of data , number of epoch and number of threads
model.fit(data['train'], epochs=30, num_threads=2)

# lest now generate a recommendation from our model by creating a dictionary that carries three parameters
def samlpe_recommendation(model, data, user_ids):
    
    #number of users and items but movies in this case in training data 
    n_users, n_items = data['train'].shape
    # creating a for loop to iterate through every user ids we import and we want report for each
    for user_id in user_ids:
        
        # lightfm considers moves of 5 and above rating as positive to make the problem binary
        # movies already liked
        # tocsr()[user_id].indices is a sub array which we retrive using the indices attribute 
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        
        #we'll get the positives from the data in COMPPRESED SPARSE ROW FORMAT
        
        #to generate recommendation and store them in the score variable using the predictive method of our model
        # movies our  model predicts they will like
        
        # user id as d 1st parameter then list of each movie the numpy gives us movies from zero to the number of items so we can predict the score of each item
        # then store them in order of their score
        scores = model.predict(user_id, np.arange(n_items))
        
        # rank them in order of most liked to least liked
        top_items = data['item_labels'][np.argsort(-scores)]
        # lets print the user id
        print("User %s" % user_id)
        print("   known positives:")
        #top 3 known positive movies the user picked
        for x in known_positives[:3]:
            #creating a for loop ending in the third index
            print("           %s" % x)
        
        # creating our recommendation and top tree movies our model predicts
        print("   Recommendation:")
        
        for x in top_items[:3]:
            print("         %s" % x)
            
           # lets use a ransom number of uder ids 
samlpe_recommendation(model, data, [3, 25, 450])

#so from here it can deployed to a website
