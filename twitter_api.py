import twitter
import tensorflow as tf
import numpy as np

api = twitter.Api(
	consumer_key="veECt8TmN6XjOSPUf8Tbs9ACi",
	consumer_secret="Y7OYrgBOjYgT6adoezc5Yzncy07EyJKezKLEDu88i3Obw39EmY",
	access_token_key="2922124038-CkJDWlQwfAug5PirV5x8PiXbflyNCLsERiy168B",
	access_token_secret="xNoWcY03pCTkzyIEELTGNmXxtuKZ5qXajUgXc4k7tRkBJ"
	)
query = "q=bitcoin OR ethereum%20&result_type=recent&since=2016-07-19&count=100"


def search(query):
	return api.GetSearch(raw_query=query)

def getTargetContextPairs(text):
	array = text.split(' ')
	out = [(array[i],array[i+1])for i in range(len(array))[:-1]]#forward pairs
	out.extend([(array[i],array[i-1])for i in range(len(array))[1:]])#backward pairs
	return out

def getTotalCorpus(numTweets):
	query = "q=bitcoin OR ethereum%20&result_type=recent&since=2016-07-19&count=100"
	results = search(query)
	tweets = results
	n = len(results)
	count = n
	corpus = getTextCorpus(results)
	while(count<numTweets):
		print(n)
		print(len(results))
		max_id = results[n-1].AsDict()['id']-1
		query = "q=bitcoin OR ethereum%20&result_type=recent&since_id="+str(max_id)+"&count=100"
		results = search(query)
		tweets.extend(results)
		n = len(results)
		count+=n
		corpus.update(getTextCorpus(results))
	return corpus, tweets

def getData(integerizedCorpus,tweets):
	data = []
	labels = []
	pairs = []
	for tweet in tweets:
		text = tweet.AsDict()['text']
		pairs.extend(getTargetContextPairs(text))
	for pair in pairs:
		data.append(integerizedCorpus[pair[0]])
		labels.append(integerizedCorpus[pair[1]])
	return data,labels



def getTextCorpus(tweets):
	corpus = set()
	for tweet in tweets:
		text = tweet.AsDict()['text']
		words = text.split(' ')
		corpus.update(words)
	return corpus

def integerizeCorpus(numTweets):
	corpus, tweets = getTotalCorpus(numTweets)
	d = {}
	count = 0
	for word in corpus:
		d[word]=count
		count+=1
	return d,corpus,tweets








