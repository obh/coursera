
# coding: utf-8

# #Using deep features to build an image classifier
# 
# #Fire up GraphLab Create

# In[6]:

import graphlab


# #Load a common image analysis dataset
# 
# We will use a popular benchmark dataset in computer vision called CIFAR-10.  
# 
# (We've reduced the data to just 4 categories = {'cat','bird','automobile','dog'}.)
# 
# This dataset is already split into a training set and test set.  

# In[5]:

image_train = graphlab.SFrame('image_train_data/')
image_test = graphlab.SFrame('image_test_data/')


# #Exploring the image data

# In[34]:

graphlab.canvas.set_target('ipynb')


# In[9]:

image_train[ 'label' ].sketch_summary()
dog_train = image_train.filter_by( [ 'dog' ], 'label')
cat_train = image_train.filter_by( [ 'cat' ], 'label')
bird_train = image_train.filter_by( [ 'bird' ], 'label')
car_train = image_train.filter_by( [ 'automobile' ], 'label')


# In[10]:

image_train['label'].sketch_summary()


# In[11]:

dog_model = graphlab.nearest_neighbors.create(dog_train, features = [ 'deep_features' ])
cat_model = graphlab.nearest_neighbors.create(cat_train,  features = [ 'deep_features' ])
bird_model = graphlab.nearest_neighbors.create(bird_train,  features = [ 'deep_features' ])
car_model = graphlab.nearest_neighbors.create(car_train,  features = [ 'deep_features' ])


# #Train a classifier on the raw image pixels
# 
# We first start by training a classifier on just the raw pixels of the image.

# In[12]:

print(dog_model.query(image_test[0:1]))
print(cat_model.query(image_test[0:1]))
print(bird_model.query(image_test[0:1]))
print(car_model.query(image_test[0:1]))


# In[13]:

print( sum(cat_model.query(image_test[0:1])['distance'])/5 )

print( sum(dog_model.query(image_test[0:1])['distance'])/5 )


# In[5]:

raw_pixel_model = graphlab.logistic_classifier.create(image_train,target='label',
                                              features=['image_array'])


# #Make a prediction with the simple model based on raw pixels

# In[14]:

dog_test = image_test.filter_by( [ 'dog' ], 'label')
cat_test = image_test.filter_by( [ 'cat' ], 'label')
bird_test = image_test.filter_by( [ 'bird' ], 'label')
car_test = image_test.filter_by( [ 'automobile' ], 'label')


# In[17]:

dog_cat_neighbors = cat_model.query(dog_test, k = 1)
dog_dog_neighbors = dog_model.query(dog_test, k = 1)
dog_bird_neighbors = bird_model.query(dog_test, k = 1)
dog_car_neighbors = car_model.query(dog_test, k = 1)


# In[19]:

dog_distances = graphlab.SFrame( 
    { 'dog_distance' : dog_dog_neighbors['distance'], 
     'cat_distance' : dog_cat_neighbors['distance'],
     'bird_distance' : dog_bird_neighbors[ 'distance' ],
     'car_distance' : dog_car_neighbors[ 'distance' ] })


# In[20]:

## this will test how many of the classifications were correct
model_result = dog_distances.apply(lambda x : 1 if x['dog_distance'] < x['cat_distance'] and x['dog_distance'] < x['bird_distance'] 
                    and x['dog_distance'] < x ['car_distance'] else 0)


# In[44]:

# accuracy of the model
sum(model_result)


# In[8]:

raw_pixel_model.predict(image_test[0:3])


# The model makes wrong predictions for all three images.

# #Evaluating raw pixel model on test data

# In[9]:

raw_pixel_model.evaluate(image_test)


# The accuracy of this model is poor, getting only about 46% accuracy.

# #Can we improve the model using deep features
# 
# We only have 2005 data points, so it is not possible to train a deep neural network effectively with so little data.  Instead, we will use transfer learning: using deep features trained on the full ImageNet dataset, we will train a simple model on this small dataset.

# In[10]:

len(image_train)


# ##Computing deep features for our images
# 
# The two lines below allow us to compute deep features.  This computation takes a little while, so we have already computed them and saved the results as a column in the data you loaded. 
# 
# (Note that if you would like to compute such deep features and have a GPU on your machine, you should use the GPU enabled GraphLab Create, which will be significantly faster for this task.)

# In[11]:

#deep_learning_model = graphlab.load_model('http://s3.amazonaws.com/GraphLab-Datasets/deeplearning/imagenet_model_iter45')
#image_train['deep_features'] = deep_learning_model.extract_features(image_train)


# As we can see, the column deep_features already contains the pre-computed deep features for this data. 

# In[12]:

image_train.head()


# #Given the deep features, let's train a classifier

# In[13]:

deep_features_model = graphlab.logistic_classifier.create(image_train,
                                                         features=['deep_features'],
                                                         target='label')


# #Apply the deep features model to first few images of test set

# In[42]:

#computing answers for the quiz
cat_train[181:182]['image'].show()
dog_train[159:160]['image'].show()


# In[15]:

deep_features_model.predict(image_test[0:3])


# The classifier with deep features gets all of these images right!

# #Compute test_data accuracy of deep_features_model
# 
# As we can see, deep features provide us with significantly better accuracy (about 78%)

# In[16]:

deep_features_model.evaluate(image_test)

