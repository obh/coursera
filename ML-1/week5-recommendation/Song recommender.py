
# coding: utf-8

# #Building a song recommender
# 
# 
# #Fire up GraphLab Create

# In[1]:

import graphlab


# #Load music data

# In[3]:

song_data = graphlab.SFrame('song_data.gl/')


# #Explore data
# 
# Music data shows how many times a user listened to a song, as well as the details of the song.

# In[4]:

song_data.head()


# ##Showing the most popular songs in the dataset

# In[5]:

graphlab.canvas.set_target('ipynb')


# In[ ]:

song_data['song'].show()


# In[ ]:

len(song_data)


# ##Count number of unique users in the dataset

# In[14]:

users = song_data['user_id'].unique()

for artist in [ "Kanye West", 'Foo Fighters', 'Taylor Swift', 'Lady GaGa']:
    print(len(song_data.filter_by( [ artist ], "artist" )['user_id'].unique()))
    


# In[31]:

import graphlab.aggregate as agg
artist_list = [ "Kanye West", 'Foo Fighters', 'Taylor Swift', 'Lady GaGa']
song_count = song_data.groupby( key_columns = 'artist', operations = {'num_songs':agg.SUM('listen_count') })
song_count.sort('num_songs', ascending=True)


# In[ ]:

len(users)


# #Create a song recommender

# In[19]:

train_data,test_data = song_data.random_split(.8,seed=0)


# ##Simple popularity-based recommender

# In[20]:

popularity_model = graphlab.popularity_recommender.create(train_data,
                                                         user_id='user_id',
                                                         item_id='song')


# ###Use the popularity model to make some predictions
# 
# A popularity model makes the same prediction for all users, so provides no personalization.

# In[ ]:

popularity_model.recommend(users=[users[0]])


# In[ ]:

popularity_model.recommend(users=[users[1]])


# ##Build a song recommender with personalization
# 
# We now create a model that allows us to make personalized recommendations to each user. 

# In[24]:

personalized_model = graphlab.item_similarity_recommender.create(train_data,
                                                                user_id='user_id',
                                                                item_id='song')


# ###Applying the personalized model to make song recommendations
# 
# As you can see, different users get different recommendations now.

# In[ ]:

personalized_model.recommend(users=[users[0]])


# In[ ]:

personalized_model.recommend(users=[users[1]])


# ###We can also apply the model to find similar songs to any song in the dataset

# In[ ]:

personalized_model.get_similar_items(['With Or Without You - U2'])


# In[ ]:

personalized_model.get_similar_items(['Chan Chan (Live) - Buena Vista Social Club'])


# #Quantitative comparison between the models
# 
# We now formally compare the popularity and the personalized models using precision-recall curves. 

# In[ ]:

if graphlab.version[:3] >= "1.6":
    model_performance = graphlab.compare(test_data, [popularity_model, personalized_model], user_sample=0.05)
    graphlab.show_comparison(model_performance,[popularity_model, personalized_model])
else:
    get_ipython().magic(u'matplotlib inline')
    model_performance = graphlab.recommender.util.compare_models(test_data, [popularity_model, personalized_model], user_sample=.05)


# The curve shows that the personalized model provides much better performance. 

# In[21]:

item_similarity_model = graphlab.item_similarity_recommender.create(train_data, user_id='user_id', item_id='song_id')


# In[22]:

subset_test_users = test_data['user_id'].unique()[0:10000]


# In[25]:

personalized_model.recommend(subset_test_users,k=1)


# In[28]:

recommendations = item_similarity_model.recommend(subset_test_users, k=1)
recommendations_count = recommendations.groupby( key_columns = 'song_id', operations = {'num_songs':agg.COUNT() })


# In[33]:

recommendations_count.sort('num_songs', ascending=False)


# In[35]:

song_data.filter_by(['SOAUWYT12A81C206F1'], 'song_id')


# In[ ]:



