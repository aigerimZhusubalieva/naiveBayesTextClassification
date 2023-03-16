'''Aigerim Zhusubalieva, az2177
Assignment 4 - Na¨ıve Bayes Text Classification
Artificial Intelligence, Fall 2022'''

#!/usr/bin/python
import sys
import copy
import math
from decimal import Decimal

filename = sys.argv[1]
stopWordsFile = sys.argv[2]
N = int(sys.argv[3])
e = 0.1

#get corpus content and store it in dictionaries
def getContent(filename, N):
	trainingSet, testSet, biography = [], [], []
	file = open(filename, 'r')
	while len(trainingSet) < N:
		line = file.readline().rstrip('\n ')
		if(line!=''):
			biography.append(line.lower())
		else:
			trainingSet.append(biography)
			biography = []

	biography = []
	for i in file:
		line = i.rstrip('\n ')
		if(line!=''):
			biography.append(line.lower())
		elif len(biography)!=0:
			testSet.append(biography)
			biography = []
	testSet.append(biography)

	file.close()
	return trainingSet, testSet

#get stopwords and store them in a dcitionary
def getStopWords(filename):
	stopWords = []
	file = open(filename, 'r')
	for i in file:
		if(i!='\n'):
			line = i.rstrip('\n')
			line = line.split(' ')
			stopWords += line
	file.close()

	stopDict = {}
	for i in stopWords:
		stopDict[i] = 1

	return stopDict

#normalize the training set using stopwords and omitting non-significant words
def normalize(wset, stopWords):
	for i in range(len(wset)):
		wordList = wset[i][2].split()
		newWordList = {}
		for j in wordList:
			j = j.rstrip(',. ')
			if j not in stopWords and len(j)>2:
				newWordList[j] = 1
		wset[i][2] = newWordList
	return wset

#normalize the test set
def normalizeTestSet(testSet, stopWords, setOfWords):
	for i in range(len(testSet)):
		wordList = testSet[i][2].split()
		newWordList = {}
		for j in wordList:
			j = j.rstrip(',. ')
			if j not in stopWords and len(j)>2:
				for k in setOfWords:
					if j in setOfWords[k]:
						newWordList[j] = -1
		testSet[i][2] = newWordList
	return testSet


def countCategories(givenSet):
	#count Occ(C) for each category C in the given set
	categories = {}
	for i in givenSet:
		if i[1] not in categories:
			categories[i[1]] = 1
		else:
			categories[i[1]] = categories[i[1]] + 1

	setOfWords = copy.deepcopy(categories)
	for i in setOfWords:
		setOfWords[i] = {}

	#count the Bernoullie number, Occ(W|C) for each word W in category C in the given set
	for biography in givenSet:
		for word in biography[2]:
			if word not in setOfWords[biography[1]]:
				setOfWords[biography[1]][word] = 1
			else:
				setOfWords[biography[1]][word] = setOfWords[biography[1]][word] + 1
	return categories, setOfWords				

#compute Freq(C) = Occ(C)/|T|
def catFreq(categories, trainingSet):
	freq = copy.deepcopy(categories)
	for i in freq:
		freq[i] = categories[i]/len(trainingSet)
	return freq

#compute Freq(W|C) = Occ(W|C)/|T|
def wordFreq(setOfWords, categories):
	freq = copy.deepcopy(setOfWords)
	allWords = {}
	for i in setOfWords:
		for word in setOfWords[i]:
			allWords[word] = 0
	for cat in freq:
		for word in setOfWords[cat]:
			freq[cat][word] = setOfWords[cat][word]/categories[cat]
		for word in allWords:
			if word not in freq[cat]:
				freq[cat][word] = 0
	return freq

#Compute the probabilities using Laplacian Correction
#P(C) = (Freq(C) + e)/(1+|C|*e)
def laplacianCorrectionCat(catFreq, e):
	for i in catFreq:
		catFreq[i] = (catFreq[i]+e)/(1+len(catFreq)*e)
	return catFreq

#P(W|C) = (Freq(W|C) + e)/(1+2*e)
def laplacianCorrectionWord(wordFreq, e):
	for i in wordFreq:
		for word in wordFreq[i]:
			wordFreq[i][word] = (wordFreq[i][word]+e)/(1+2*e)
	return wordFreq

#compute negative log probabilities
def negLogCat(catFreq):
	for i in catFreq:
		catFreq[i] = math.log(catFreq[i], 2)*(-1)
	return catFreq

def negLogWord(wordFreq):
	for i in wordFreq:
		for word in wordFreq[i]:
			wordFreq[i][word] = math.log(wordFreq[i][word], 2)*(-1)
	return wordFreq

#train the program
def train(trainingSet, stopWords, e):
	catFreqv, wordFreqv = {}, {}
	trainingSet = normalize(trainingSet, stopWords)
	categories, setOfWords = countCategories(trainingSet)
	catFreqv = catFreq(categories, trainingSet)
	catFreqv = laplacianCorrectionCat(catFreqv, e)
	catFreqv = negLogCat(catFreqv)
	wordFreqv = wordFreq(setOfWords, categories)
	wordFreqv = laplacianCorrectionWord(wordFreqv, e)
	wordFreqv = negLogWord(wordFreqv)
	return catFreqv, wordFreqv

#run the test corpus through the trained program
def test(tesSet, catFreq, wordFreq):
	#predicted category of biographies
	pred = {}
	#probabilities of each category for all biographies
	probAll = {}

	for bio in testSet:
		cats = {}
		#compute L(C|B) = L(C)+sum of all L(W|C) for each biography B
		for cat in catFreq:
			cats[cat] = catFreq[cat]
			for word in bio[2]:
				cats[cat] = cats[cat] + wordFreq[cat][word]

		#make the prediction
		minKey = min(cats, key=cats.get)
		minVal = cats[min(cats, key=cats.get)]
		
		pred[bio[0]] = [minKey, minVal]

		#compute P(Ck|B) = xi/s where s = sum of all xi, xi = 2^(m-ci), ci = L(Ci|B), m = min(ci) for each category C
		x = {}
		s = 0
		for i in cats:
			if cats[i]-minVal < 7:
				x[i] = 2**(minVal-cats[i])
			else:
				x[i] = 0
			s += x[i]

		prob = {}
		for i in cats:
			prob[i] = round(x[i]/s, 2)
		
		probAll[bio[0]] = prob

	return pred, probAll

def printResults(pred, prob, testSet):
	correct = 0
	for bio in testSet:
		str1 = bio[0].title() + ".   " + " Prediction: " + pred[bio[0]][0].title() + "."
		if(pred[bio[0]][0] == bio[1]):
			correct+=1
			str1 = str1 + "   Right.\n"
		else:
			str1 = str1 + "   Wrong.\n"
		
		for category in prob[bio[0]]:
			str1 += category.title() + ": " + str(prob[bio[0]][category]) + "   "

		print(str1, '\n')
		print("Overall accuracy:", correct, "out of", len(testSet), "=", round(correct/len(testSet), 2))

#training and test sets are list of [name, category, dictionary of words in biography]
trainingSet, testSet = getContent(filename, N)
#stopwords is a dictionary of stopwords as keys
stopWords = getStopWords(stopWordsFile)
#catFreq is a dictionary with category:frequency
#wordFreq is a dictionary with category:{word:frequency}
catFreq, wordFreq = train(trainingSet, stopWords, e)
testSet = normalizeTestSet(testSet, stopWords, wordFreq)
#pred is a dicrionary with name: [pred category, the L value]
#prob is a dictionary with name: {category:prob of that category}
pred, prob = test(testSet, catFreq, wordFreq)
printResults(pred, prob, testSet)



















