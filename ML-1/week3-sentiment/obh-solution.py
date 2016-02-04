#!/usr/bin/python

products = graphlab.SFrame('amazon_baby.gl/')

selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']

products[ 'word_count'  ] = graphlab.text_analytics.count_words(products['review'])
products[ 'selected_words' ]  = products['word_count'].apply(lambda x: { y : x[y] if y in x else 0 for y in selected_words  } )

# solution to first two questions
q1 = { word : sum( item[word] for item in products['selected_words']  ) for word in selected_words }

products = products[products['rating'] != 3]
products[ 'sentiment'  ] = products[ 'rating'  ] >= 4

train_data,test_data = products.random_split(.8, seed=0)
selected_words_model = graphlab.logistic_classifier.create(train_data,target='sentiment', features=['selected_words'], validation_set=test_data)
selected_words_model['coefficients']

## solutions to quiz questions

selected_words_model['coefficients'].sort( 'value'  )

selected_words_model.evaluate(test_data)

diaper_champ = products[products['name'] == 'Baby Trend Diaper Champ']

diaper_champ['predicted_sentiment'] = sentiment_model.predict(diaper_champ, output_type='probability')

diaper_champ.sort('predicted_sentiment', ascending = False)

selected_word_sentiment = selected_words_model.predict(diaper_champ[0:1], output_type = 'probability')

