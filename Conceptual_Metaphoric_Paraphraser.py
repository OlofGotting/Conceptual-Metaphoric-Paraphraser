import numpy as np
import math
import nltk
import pickle
import re
import pandas as pd
import csv
from nltk.corpus import framenet as fn, wordnet as wn
from nltk.stem import WordNetLemmatizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
from collections import Counter, defaultdict, OrderedDict
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.spatial import distance
from operator import itemgetter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from lm_scorer.models.auto import AutoLMScorer as LMScorer

sentence_vector_model = SentenceTransformer('paraphrase-distilroberta-base-v1')

class Knowledge:

	# Knowledge representation to allow for reasoning about metaphoric paraphrasing.

	def __init__(self):
		with open('framedict', 'rb') as framedict, \
			open('allmetaphors', 'rb') as metaphorlist, \
			open('metaphordict', 'rb') as metaphordict, \
			open('lemposdict', 'rb') as lemposdict, \
			open('sentimentdict_float_keys', 'rb') as sentimentdict_float_keys, \
			open('sentimentdict', 'rb') as sentimentdict2, \
			open('dictoftargetsynonyms', 'rb') as target_synonym_dict, \
			open('setofallnouns', 'rb') as setofallnouns, \
			open('dictof500adjectives', 'rb') as dictof500adjectives, \
			open('setof500adjectives', 'rb') as setof500adjectives:

			self.framedict = pickle.load(framedict) # frame relationships from metanet
			self.metaphors = pickle.load(metaphorlist) # list of metaphors from metanet
			self.metaphordict = pickle.load(metaphordict) # dict of metaphors, key == target, value == source
			self.sentimentdict_float_keys = pickle.load(sentimentdict_float_keys) # dict of adjectives and their sentiments from saifmohammad.com
			self.target_synonym_dict = pickle.load(target_synonym_dict)
			self.lemposdict = pickle.load(lemposdict) # dict of words and their hyponyms from wordnet. (to create more variation and perceived creativity)
			self.setofallnouns = pickle.load(setofallnouns)
			self.setof500adjectives = pickle.load(setof500adjectives)
			self.dictof500adjectives = pickle.load(dictof500adjectives)
			self.sentimentdict2 = pickle.load(sentimentdict2)

		self.robertamodel = SentenceTransformer('paraphrase-distilroberta-base-v1')
		self.sid = SentimentIntensityAnalyzer()
		self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
		self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
		self.scorer = LMScorer.from_pretrained("gpt2", device='cpu', batch_size=1)

	

class Paraphrase:

	'''The idea here is the following:
		An input in the form of "<Noun phrase> is <Adjective phrase>" is taken.
		Possible metaphoric paraphrases of the form "<Noun phrase> is (<adjective>)<noun phrase>" are generated.
		The paraphrase candidates are ranked by these feature values:
			1. Cosine distance between the embeddings of the paraphrase/input sentence.
			2. How well the sentiment of the inputs adjective phrase matches that of the paraphrase candidates.
			3. Difference in surprisal, as measured by difference in gpt2 log probability. 
		The script prints the three top ranked sentences and their score for each feature value.

'''
	def __init__(self, knowledge_object):
		self.model = knowledge_object
		

	def start(self, csv_file):
		with open(csv_file, 'r') as csv_in:
			self.csv_file = csv_in.read()

		self.logreg = LogReg(csv_file)
		self.insentences = []
		
		for element in self.csv_file.split('\n'):
			line = element.replace('\n', '').replace('"', '').replace('ï', '').replace('»', '').replace('¿', '').split(';')
			self.insentences.append(line[0])

		self.cosine_weight = self.find_weights(self.csv_file)[0]
		self.sentdif_weight = self.find_weights(self.csv_file)[1]
		self.sentprob_weight = self.find_weights(self.csv_file)[2]

		for sentence in self.insentences:
			print(sentence)
			self.insentence = sentence
			self.prefix = self.insentence.split()[0]
			self.split_sentence = self.insentence.strip(',').replace(' and', '').split()[1:]
			self.insentence_sentiment = self.model.sid.polarity_scores(self.insentence)['compound']

			if 'is' in self.split_sentence:
				self.are = 'is'
				self.is_index = self.split_sentence.index('is')
			elif 'are' in self.split_sentence:
				self.are = 'are'
				self.is_index = self.split_sentence.index('are')
			elif 'was' in self.split_sentence:
				self.are = 'was'
				self.is_index = self.split_sentence.index('was')
			elif 'were' in self.split_sentence:
				self.are = 'were'
				self.is_index = self.split_sentence.index('were')

			self.original_adjective_phrase = ' '.join(self.split_sentence[self.is_index + 1:])
			self.original_adjective_phrase_sentiment = self.model.sid.polarity_scores(self.original_adjective_phrase)['compound']
			self.an = None
			self.picked_target = self.pick_target(self.split_sentence)
			self.adjective_candidates = self.find_adjective_candidates()
			self.source_candidates = self.find_source_candidates()
			self.create_paraphrase_array(self.source_candidates, self.adjective_candidates)
			self.print_best_candidate()


	def pick_target(self, split_sentence):
		noun_phrase = ' '.join(split_sentence[:self.is_index])
		return noun_phrase

	def find_source_candidates(self):

		source_candidates = set()
		for targetframe in self.model.target_synonym_dict[self.picked_target]:

			if targetframe in self.model.metaphordict.keys():
				
				for frame in self.model.metaphordict[targetframe]:
					if frame != '':
						source_candidates.add(frame.lower())

					for word in self.model.lemposdict[frame]:
						if word == self.split_sentence[0]:
							pass
						elif word == '':
							pass
						else:
							if self.check_if_noun(word) != None:
								source_candidates.add(word.replace('.v', '').replace('.n', '').replace('.a', '').replace('.n', '').lower())

		return(source_candidates)


	def check_if_noun(self, phrase):
		a = wn.synsets(phrase, pos=wn.NOUN)
		if len(phrase.split()) > 1:
			return phrase
		else:
			if len(a) > 0:
				return phrase
			else:
				return None

	def find_adjective_candidates(self):
		adjective_candidates = set()
		lower_limit = self.original_adjective_phrase_sentiment - 0.2
		upper_limit = self.original_adjective_phrase_sentiment + 0.2
		list_of_keys = sorted(list(self.model.dictof500adjectives.keys()))
		closest_lower = min(range(len(list_of_keys)), key=lambda i: abs(list_of_keys[i]-lower_limit))
		closest_upper = min(range(len(list_of_keys)), key=lambda i: abs(list_of_keys[i]-upper_limit))
		relevant_adjectives = list_of_keys[closest_lower:closest_upper]
		for n in relevant_adjectives:
			for element in self.model.dictof500adjectives[n]:
				if element not in self.insentence.replace(',', '').replace('.', '').split():
					adjective_candidates.add(element)
		return adjective_candidates

	def generate_sentence(self, noun_phrase):
		if noun_phrase[0][0] in 'aeiou':
			self.an = 'an'
		elif self.are in ['are', 'were']:
			self.an = '' #catches plural
		else:
			self.an = 'a'

		self.outsentence = ' '.join([self.prefix, self.picked_target, self.are, self.an, ' '.join([word for word in noun_phrase])])
		self.outsentence = self.outsentence.replace('  ', ' ')
		if self.outsentence.endswith('.'):
			return self.outsentence
		else:
			return self.outsentence + '.'

	def find_weights(self, csv_file):
		return self.logreg.big_result


	def check_cosine_distance(self, sentence_list):
		sentence_embeddings = self.model.robertamodel.encode(sentence_list)
		return distance.cosine(sentence_embeddings[0], sentence_embeddings[1]) 

	def check_sentiment_difference(self, sentence_list):
		if self.model.sid.polarity_scores(sentence_list[0])['compound'] == 0 \
		and self.model.sid.polarity_scores(sentence_list[0])['neg'] == 0 \
		and self.model.sid.polarity_scores(sentence_list[0])['neu'] == 0 \
		and self.model.sid.polarity_scores(sentence_list[0])['pos'] == 0:
			return None

		elif self.model.sid.polarity_scores(sentence_list[1])['compound'] == 0 \
			and self.model.sid.polarity_scores(sentence_list[1])['neg'] == 0 \
			and self.model.sid.polarity_scores(sentence_list[1])['neu'] == 0 \
			and self.model.sid.polarity_scores(sentence_list[1])['pos'] == 0:
			return None

		else:
			sentiment_difference = (self.model.sid.polarity_scores(sentence_list[0])['compound'] - self.model.sid.polarity_scores(sentence_list[1])['compound'])
			return sentiment_difference

	def check_sentence_probability(self, sentence_list):
		log_p_candidate = self.model.scorer.sentence_score(sentence_list[1], log=True)
		log_p_insentence = self.model.scorer.sentence_score(sentence_list[0], log=True)
		feature_value = (log_p_candidate - log_p_insentence) / len(sentence_list[0])
		return feature_value

	def create_paraphrase_array(self, source_candidates, adjective_candidates):
		np.set_printoptions(suppress=True)
		self.potentials_list = []
		self.big_paraphrase_array = np.empty((0, 3))
		self.small_paraphrase_array = np.empty((0, 3))
		self.small_sentence_list = []
		self.big_sentence_list = []

		for indx, source in enumerate(source_candidates):
			if source == self.split_sentence[0]:
				pass
			sentence_to_check = self.generate_sentence([source])
			self.populate_array(sentence_to_check, 'big')
		self.big_predicted = self.logreg.big_clf.predict_proba(self.big_paraphrase_array) 
		toplist = []

		for x, y in enumerate(self.big_predicted):
			toplist.append((x, y[1])) # (rank, totalrating)
		toplist = sorted(toplist, key=itemgetter(1), reverse=True)
		

		for rank, tup in enumerate(toplist[:10]):
			to_append = []
			candidate = self.potentials_list[toplist[rank][0]]
			split_candidate = candidate.split()
			source = ' '.join(split_candidate[self.is_index + 3:])
			print('source: ', source, rank + 1, '/', 10)
			sentence_to_check = self.generate_sentence([source])
			self.populate_array(sentence_to_check, 'small')

			for adjective in adjective_candidates:
				sentence_to_check = self.generate_sentence([adjective, source])
				self.populate_array(sentence_to_check, 'small')
		self.small_predicted = self.logreg.big_clf.predict_proba(self.small_paraphrase_array)

	def populate_array(self, sentence_to_check, array_to_modify):
		
		to_append = []
		
		if self.check_sentiment_difference([self.insentence, sentence_to_check]) == None:
			number_of_sentences_where_vader_failed += 1
			pass
		
		else:
			self.potentials_list.append(sentence_to_check)
			to_append.append(self.check_cosine_distance([self.insentence, sentence_to_check])) #* self.cosine_weight
			to_append.append(self.check_sentiment_difference([self.insentence, sentence_to_check]))# * self.sentdif_weight
			to_append.append(self.check_sentence_probability([self.insentence, sentence_to_check]))# * self.sentprob_weight
			
			if array_to_modify == 'big':
				self.big_paraphrase_array = np.append(self.big_paraphrase_array, np.array([to_append]), axis=0)
				self.big_sentence_list.append(sentence_to_check)
			
			elif array_to_modify == 'small':
				self.small_paraphrase_array = np.append(self.small_paraphrase_array, np.array([to_append]), axis=0)
				self.small_sentence_list.append(sentence_to_check)

	def print_best_candidate(self):

		maxindex = np.argmax(self.small_predicted[:,1])
		print(self.small_predicted[maxindex,:])
		print(self.small_sentence_list[maxindex])
		print('cosine distance: ', self.small_paraphrase_array[maxindex,0])
		print('sentiment difference : ', self.small_paraphrase_array[maxindex,1])
		print('sentence probability: ', self.small_paraphrase_array[maxindex,2])

		with open('meta1_in_and_out.txt', 'a') as f:
			print(self.insentence.capitalize(), '/', self.small_sentence_list[maxindex].capitalize(), file = f)
		with open('meta1_out.txt', 'a') as f:
			print(self.small_sentence_list[maxindex].capitalize(), file = f)

class LogReg:

	'''Logistic regression performed on a .csv file to weight the feature values.
	   The .csv file has three columns:
	   1. Original literal sentences.
	   2. Metaphoric paraphrases of the original sentences written by a native speaker of English. (are given the value 1)
	   3. Randomly generated (and thus pretty bad) paraphrases of the orignal sentences. (are given the value 0)
	   '''

	def __init__(self, csv_file):
		with open(csv_file, 'r') as csv_file:
			self.csv_file = csv_file
			self.paraphraseobject = Paraphrase(knowledge_representation)
			self.dfs = self.make_pandas_frame(self.csv_file)
			self.small_df = self.dfs[0]
			self.big_df = self.dfs[1]
			self.small_clf = self.logistic_regression(self.small_df)
			self.big_clf = self.logistic_regression(self.big_df) 
			self.small_result = self.small_clf.coef_[0]
			self.big_result = self.big_clf.coef_[0]

	def make_pandas_frame(self, csv_file):
		small_df = pd.DataFrame(columns = ('codist','sentdif', 'good_or_not'))
		big_df = pd.DataFrame(columns = ('codist','sentdif', 'sentprob', 'good_or_not'))
		insentences = []
		good_paraphrases = []
		bad_paraphrases = []
		goodparaphraseindex = -1
		badparaphraseindex = 29

		for line in csv_file:
			data = line.replace('\n', '').replace('"', '').replace('ï', '').replace('»', '').replace('¿', '').split(';')
			insentences.append(data[0])
			good_paraphrases.append(data[1])
			bad_paraphrases.append(data[2])

		for insentence, good_sentence in zip(insentences, good_paraphrases):

			goodparaphraseindex += 1
			codist = self.paraphraseobject.check_cosine_distance([insentence, good_sentence])
			sentdif = self.paraphraseobject.check_sentiment_difference([insentence, good_sentence])
			sentprob = self.paraphraseobject.check_sentence_probability([insentence, good_sentence])
			zero_or_one = 1
			small_df.loc[goodparaphraseindex] = [codist, sentdif, int(1)]
			big_df.loc[goodparaphraseindex] = [codist, sentdif, sentprob, int(1)]


		for insentence, bad_sentence in zip(insentences, bad_paraphrases):

			badparaphraseindex += 1
			codist = self.paraphraseobject.check_cosine_distance([insentence, bad_sentence])
			sentdif = self.paraphraseobject.check_sentiment_difference([insentence, bad_sentence])
			sentprob = self.paraphraseobject.check_sentence_probability([insentence, bad_sentence])
			zero_or_one = 0
			small_df.loc[badparaphraseindex] = [codist, sentdif, int(0)]
			big_df.loc[badparaphraseindex] = [codist, sentdif, sentprob, int(0)]

		return [small_df, big_df] #_normalized


	def logistic_regression(self, dataframe):

		X = dataframe.drop('good_or_not', axis=1)
		y = dataframe['good_or_not']
		clf = LogisticRegression(random_state=0).fit(X, y)
		return clf


if __name__ == '__main__':
	knowledge_representation = Knowledge()
	paraphrase = Paraphrase(knowledge_representation)
	paraphrase.start('sentences_max2.csv')
