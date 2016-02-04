#!/usr/bin/python

import graphlab

people = graphlab.SFrame('people_wiki.gl/')

elton_john = people[people['name'] == 'Elton John']
elton_john['word_count'] = graphlab.text_analytics.count_words(elton_john['text'])
elton_john_word_cnt_table = elton_john[['word_count']].stack('word_count', new_column_name = ['word','count'])
elton_john_word_cnt_table.sort('count', ascending = False)
people['word_count'] = graphlab.text_analytics.count_words(people['text'])
tfidf = graphlab.text_analytics.tf_idf( people[ 'word_count'  ] )
people[ 'tfidf'  ] = tfidf
elton_john_tfidf = people[people['name'] == 'Elton John']
elton_john_tfidf = elton_john_tfidf[['tfidf']].stack('tfidf', new_column_name = [ 'word', 'tfidf' ] )
elton_john_tfidf.sort( 'tfidf', ascending = False )

victoria_beckham = people[people['name'] == 'Victoria Beckham']
paul_mccartney = people[people['name'] == 'Paul McCartney']

print(graphlab.distances.cosine(elton_john['tfidf'][0], victoria_beckham['tfidf'][0]))
print(graphlab.distances.cosine(elton_john['tfidf'][0], paul_mccartney['tfidf'][]))

knn_tfidf_model = graphlab.nearest_neighbors.create(people,features=['tfidf'],label='name', distance = "cosine")
knn_wordcount_model = graphlab.nearest_neighbors.create(people,features=['word_count'],label='name', distance = "cosine")

print( knn_tfidf_model.query(elton_john)  )
print( knn_wordcount_model.query(elton_john)  )

print( knn_tfidf_model.query(victoria_beckham)  )
print( knn_wordcount_model.query(victoria_beckham)  )
