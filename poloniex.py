import urllib
import urllib2
import json
import time
import hmac,hashlib
import os
import time
import pandas as pd
import pdb

key = 'Z3FRTM17-3N17VQWH-L9W3PWMF-T4MC7JNW'
secret = 'c65c99a9a94b034fc19569170308bc29c86eacc539fa5bf7b5d34714dce0c1ac4fd60309c710d0ddac7e8b6605e868ddaf557b9745442b50c30ad98d0c94be62'

FETCH_URL = "https://poloniex.com/public?command=returnChartData&currencyPair=%s&start=%d&end=%d&period=300"
TICKER_URL = "https://poloniex.com/public?command=returnTicker"
#PAIR_LIST = ["BTC_ETH"]
DATA_DIR = "realtime_data"
COLUMNS = ["date","high","low","open","close","volume","quoteVolume","weightedAverage"]
PAIRS = ["BTC_BTS","BTC_ZEC","BTC_STRAT","BTC_XEM","BTC_STEEM","BTC_LTC","BTC_ETC","BTC_XRP","BTC_XMR","BTC_DASH","BTC_ETH"]
PAIRS_DICT = {0:"BTC_BTS",1:"BTC_ZEC",2:"BTC_STRAT",3:"BTC_XEM",4:"BTC_STEEM",5:"BTC_LTC",6:"BTC_ETC",7:"BTC_XRP",8:"BTC_XMR",9:"BTC_DASH",10:"BTC_ETH"}
PARAMS = ['high','open','volume']


def get_new_portfolio(cur_portfolio,new_portfolio):
    df = get_current_rates()
    differences = new_portfolio-cur_portfolio

def get_current_rates():
    df = pd.read_json(TICKER_URL, convert_dates=False)
    return df

def fetch_data(start_time=1512259200,end_time=9999999999):
    path = DATA_DIR
    for pair in PAIRS:
        datafile = os.path.join(path, pair+"_test"+".csv")
        timefile = os.path.join(path, pair+"_test")
        url = FETCH_URL % (pair, start_time, end_time)
        df = pd.read_json(url, convert_dates=False)
        if df["date"].iloc[-1] == 0:
            print("No data.")
        end_time = df["date"].iloc[-1]
        #pdb.set_trace()
        ft = open(timefile,"w+")
        ft.write("%d\n" % end_time)
        ft.close()
        outf = open(datafile, "a")
        df.to_csv(outf, index=False, columns=COLUMNS)
        outf.close()
        print("Finish.")
        #time.sleep(30)

def get_data(pair):
    datafile = os.path.join(DATA_DIR, pair+".csv")
    timefile = os.path.join(DATA_DIR, pair)

    if os.path.exists(datafile):
        newfile = False
        start_time = int(open(timefile).readline()) + 1
    else:
        newfile = True
        start_time = 1388534400     # 2014.01.01
    end_time = 9999999999#start_time + 86400*30

    url = FETCH_URL % (pair, start_time, end_time)
    print("Get %s from %d to %d" % (pair, start_time, end_time))

    df = pd.read_json(url, convert_dates=False)

    #import pdb;pdb.set_trace()

    if df["date"].iloc[-1] == 0:
        print("No data.")
        return

    end_time = df["date"].iloc[-1]
    ft = open(timefile,"w")
    ft.write("%d\n" % end_time)
    ft.close()
    outf = open(datafile, "a")
    if newfile:
        df.to_csv(outf, index=False, columns=COLUMNS)
    else:
        df.to_csv(outf, index=False, columns=COLUMNS, header=False)
    outf.close()
    print("Finish.")
    time.sleep(30)
 
def createTimeStamp(datestr, format="%Y-%m-%d %H:%M:%S"):
    return time.mktime(time.strptime(datestr, format))
 
class poloniex:
    def __init__(self, APIKey, Secret):
        self.APIKey = APIKey
        self.Secret = Secret
 
    def post_process(self, before):
        after = before
 
        # Add timestamps if there isnt one but is a datetime
        if('return' in after):
            if(isinstance(after['return'], list)):
                for x in xrange(0, len(after['return'])):
                    if(isinstance(after['return'][x], dict)):
                        if('datetime' in after['return'][x] and 'timestamp' not in after['return'][x]):
                            after['return'][x]['timestamp'] = float(createTimeStamp(after['return'][x]['datetime']))
                           
        return after
 
    def api_query(self, command, req={}):
 
        if(command == "returnTicker" or command == "return24Volume"):
            ret = urllib2.urlopen(urllib2.Request('https://poloniex.com/public?command=' + command))
            return json.loads(ret.read())
        elif(command == "returnOrderBook"):
            ret = urllib2.urlopen(urllib2.Request('https://poloniex.com/public?command=' + command + '&currencyPair=' + str(req['currencyPair'])))
            return json.loads(ret.read())
        elif(command == "returnMarketTradeHistory"):
            ret = urllib2.urlopen(urllib2.Request('https://poloniex.com/public?command=' + "returnTradeHistory" + '&currencyPair=' + str(req['currencyPair'])))
            return json.loads(ret.read())
        else:
            req['command'] = command
            req['nonce'] = int(time.time()*1000)
            post_data = urllib.urlencode(req)
 
            sign = hmac.new(self.Secret, post_data, hashlib.sha512).hexdigest()
            headers = {
                'Sign': sign,
                'Key': self.APIKey
            }
 
            ret = urllib2.urlopen(urllib2.Request('https://poloniex.com/tradingApi', post_data, headers))
            jsonRet = json.loads(ret.read())
            return self.post_process(jsonRet)
 
 
    def returnTicker(self):
        return self.api_query("returnTicker")
 
    def return24Volume(self):
        return self.api_query("return24Volume")
 
    def returnOrderBook (self, currencyPair):
        return self.api_query("returnOrderBook", {'currencyPair': currencyPair})
 
    def returnMarketTradeHistory (self, currencyPair):
        return self.api_query("returnMarketTradeHistory", {'currencyPair': currencyPair})
 
 
    # Returns all of your balances.
    # Outputs:
    # {"BTC":"0.59098578","LTC":"3.31117268", ... }
    def returnBalances(self):
        return self.api_query('returnBalances')
 
    # Returns your open orders for a given market, specified by the "currencyPair" POST parameter, e.g. "BTC_XCP"
    # Inputs:
    # currencyPair  The currency pair e.g. "BTC_XCP"
    # Outputs:
    # orderNumber   The order number
    # type          sell or buy
    # rate          Price the order is selling or buying at
    # Amount        Quantity of order
    # total         Total value of order (price * quantity)
    def returnOpenOrders(self,currencyPair):
        return self.api_query('returnOpenOrders',{"currencyPair":currencyPair})
 
 
    # Returns your trade history for a given market, specified by the "currencyPair" POST parameter
    # Inputs:
    # currencyPair  The currency pair e.g. "BTC_XCP"
    # Outputs:
    # date          Date in the form: "2014-02-19 03:44:59"
    # rate          Price the order is selling or buying at
    # amount        Quantity of order
    # total         Total value of order (price * quantity)
    # type          sell or buy
    def returnTradeHistory(self,currencyPair):
        return self.api_query('returnTradeHistory',{"currencyPair":currencyPair})
 
    # Places a buy order in a given market. Required POST parameters are "currencyPair", "rate", and "amount". If successful, the method will return the order number.
    # Inputs:
    # currencyPair  The curreny pair
    # rate          price the order is buying at
    # amount        Amount of coins to buy
    # Outputs:
    # orderNumber   The order number
    def buy(self,currencyPair,rate,amount):
        return self.api_query('buy',{"currencyPair":currencyPair,"rate":rate,"amount":amount})
 
    # Places a sell order in a given market. Required POST parameters are "currencyPair", "rate", and "amount". If successful, the method will return the order number.
    # Inputs:
    # currencyPair  The curreny pair
    # rate          price the order is selling at
    # amount        Amount of coins to sell
    # Outputs:
    # orderNumber   The order number
    def sell(self,currencyPair,rate,amount):
        return self.api_query('sell',{"currencyPair":currencyPair,"rate":rate,"amount":amount})
 
    # Cancels an order you have placed in a given market. Required POST parameters are "currencyPair" and "orderNumber".
    # Inputs:
    # currencyPair  The curreny pair
    # orderNumber   The order number to cancel
    # Outputs:
    # succes        1 or 0
    def cancel(self,currencyPair,orderNumber):
        return self.api_query('cancelOrder',{"currencyPair":currencyPair,"orderNumber":orderNumber})
 
    # Immediately places a withdrawal for a given currency, with no email confirmation. In order to use this method, the withdrawal privilege must be enabled for your API key. Required POST parameters are "currency", "amount", and "address". Sample output: {"response":"Withdrew 2398 NXT."}
    # Inputs:
    # currency      The currency to withdraw
    # amount        The amount of this coin to withdraw
    # address       The withdrawal address
    # Outputs:
    # response      Text containing message about the withdrawal
    def withdraw(self, currency, amount, address):
        return self.api_query('withdraw',{"currency":currency, "amount":amount, "address":address})

p = poloniex(key,secret)


def main():
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    df = pd.read_json("https://poloniex.com/public?command=return24hVolume")
    pairs = [pair for pair in df.columns if pair.startswith('BTC')]
    print(pairs)

    for pair in pairs:
        get_data(pair)
        time.sleep(2)

#if __name__ == '__main__':
#    main()"""