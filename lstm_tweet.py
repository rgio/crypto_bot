import tensorflow as tf
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import cPickle

embeddingsFileName = "word2vecEmbeddings.p"
dictFileName = "word2vecDict.p"
reverseDictFileName = "reverseWord2vecDict.p"
tweetsFileName = "tweets.p"
#word2vecmodel, embedding_weights, word_features, data = data_prep()
# To do tomorrow:
# 1. rename params for the alpha part
# 2. find out





class LstmTweet:

    def __init__(self):
        self.inititalize_data()
        self.input = self.processTweets()

    def inititalize_data(self):
        self.w2vEmbeddings = cPickle.load(open(embeddingsFileName))
        self.w2vDict = cPickle.load(open(dictFileName))
        self.w2vReversedDict = cPickle.load(open(reverseDictFileName))
        self.tweets = cPickle.load(open(tweetsFileName))
        self.dimEmbedding = self.w2vEmbeddings.shape[1]

    

    def processTweets(self):
        def getWordVector(word):
            print("word",word)
            key = word
            if(key not in self.w2vDict.keys()):
                key = 'UNK'
            return self.w2vEmbeddings[self.w2vDict[key]]
        maxLength = 0
        for tweet in self.tweets:
            d = tweet.AsDict()
            text = d['text']
            l = len(text.split(' '))
            if(l>maxLength):
                maxLength=l

        dim = self.dimEmbedding
        numTweets = len(self.tweets)
        #print("maxlength",maxLength)
        #print("dim",dim)
        tweetTensor = np.zeros((numTweets,maxLength,dim) )
        for j in range(numTweets):
            #print("count",count)
            tweet = self.tweets[j]
            d = tweet.AsDict()
            text = d['text']
            words = [tf.compat.as_str(w) for w in text.split(' ')]
            wordVectors = np.zeros( (maxLength,dim) )
            #print("LEN",len(words))
            for i in range(len(words)):
                print(i)
                v = getWordVector(words[i])
                wordVectors[i]=v
            count = wordVectors.shape[0]
            while(count<maxLength):
                count+=1
                wordVectors = np.vstack( (wordVectors,np.zeros((1,dim) )))
            print("WV",wordVectors)
            tweetTensor[j]=wordVectors

        return tweetTensor


"""embedding_encoder = variable_scope.get_variable(
    "embedding_encoder", [src_vocab_size, embedding_size], ...)
encoder_emb_inp = embedding_ops.embedding_lookup(
    embedding_encoder, encoder_inputs)"""

a = LstmTweet()
tweets = a.processTweets()
batch_size = len(tweets)
lengths = [len(tweet) for tweet in tweets]
tweets = tf.convert_to_tensor(tweets,np.float32)
#tweet = tweets[0]

encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(50)
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
    encoder_cell, tweets,
    sequence_length=lengths,initital_state=encoder_cell.zero_state(batch_size, tf.float32), time_major=True)




