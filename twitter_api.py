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




"""numTweets = 90
corpus, tweets = getTotalCorpus(numTweets)
int_corpus = integerizeCorpus(corpus)
data,labels = getData(int_corpus,tweets)"""




"""embeddings = tf.Variable(
	tf.random_uniform([vocabulary_size, embedding_size], -1.,1.))

nce_weights = tf.Variable(
	tf.truncated_normal([vocabulary_size, embedding_size],
                      stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size,1])
embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# Compute the NCE loss, using a sample of the negative labels each time.
loss = tf.reduce_mean(
  tf.nn.nce_loss(weights=nce_weights,
                 biases=nce_biases,
                 labels=train_labels,
                 inputs=embed,
                 num_sampled=num_sampled,
                 num_classes=vocabulary_size))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

for inputs, labels in generate_batch(...):
  feed_dict = {train_inputs: inputs, train_labels: labels}
  _, cur_loss = session.run([optimizer, loss], feed_dict=feed_dict)"""










