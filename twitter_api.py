import twitter
import tensorflow as tf

api = twitter.Api(
	consumer_key="veECt8TmN6XjOSPUf8Tbs9ACi",
	consumer_secret="Y7OYrgBOjYgT6adoezc5Yzncy07EyJKezKLEDu88i3Obw39EmY",
	access_token_key="2922124038-CkJDWlQwfAug5PirV5x8PiXbflyNCLsERiy168B",
	access_token_secret="xNoWcY03pCTkzyIEELTGNmXxtuKZ5qXajUgXc4k7tRkBJ"
	)


def search(query):
	return api.GetSearch(raw_query=query)

def getTotalCorpus(numTweets):
	query = "q=bitcoin OR ethereum%20&result_type=recent&since=2016-07-19&count=100"
	results = search(query)
	n = len(results)
	count = n
	corpus = getTextCorpus(results)
	while(count<numTweets):
		print(n)
		print(len(results))
		max_id = results[n-1].AsDict()['id']-1
		query = "q=bitcoin OR ethereum%20&result_type=recent&since_id="+str(max_id)+"&count=100"
		results = search(query)
		n = len(results)
		count+=n
		corpus.update(getTextCorpus(results))
	return corpus



def getTextCorpus(tweets):
	corpus = set()
	for tweet in tweets:
		text = tweet.AsDict()['text']
		words = text.split(' ')
		corpus.update(words)
	return corpus

numTweets = 10000
corpus = getTotalCorpus(numTweets)