#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:23:31 2017

@author: vinaya
"""

import os
import io
import re
import numpy as np

#logistic regression
class LogisticRegression:
    def __init__(self,trainHam,trainSpam, testHam, testSpam):
        self.trainHam = trainHam
        self.trainSpam = trainSpam
        self.testHam = testHam
        self.testSpam = testSpam
        self.weights = {}
        self.vocabulary = []
        self.rate=0.0001
        self.lambdaVal = 5
        self.fileData = []
        
    def run(self):
        self.createVocab()
        folder = os.listdir(self.trainHam)
        for each in folder:
            self.processFile(self.trainHam+"/"+each,1.0)
            
        folder = os.listdir(self.trainSpam)
        for each in folder:
            self.processFile(self.trainSpam+"/"+each,0.0)
        



    def countWords(self,path,wordCount):
        
        f = io.open(path, 'r',encoding='iso-8859-1')
        lines = f.readlines()
        
        for line in lines:
            lettersOnly = (re.sub("[^a-zA-Z0-9\s]", "", line)).lower().split()
            for word in lettersOnly:
                if word in wordCount:
                    wordCount[word]+=1
                else:
                    wordCount[word]=1 
        f.close()        
     
                       
    def createVocab(self):
        hamWords = {}
        folder = os.listdir(self.trainHam)
        for each in folder:
            self.countWords(self.trainHam+"/"+each,hamWords)
        
        spamWords = {}
        folder = os.listdir(self.trainSpam)
        for each in folder:
            self.countWords(self.trainSpam+"/"+each,spamWords)
        
        self.vocabulary = set(list(hamWords.keys())+list(spamWords.keys()))
        for each in self.vocabulary:
            self.weights[each] = 0.0
        
        
    def processFile(self,file,classification):
        wordCount = {}        
        self.countWords(file,wordCount)
        self.fileData.append({'fileName':file,'token':wordCount,'class':classification})
       
    def train(self):
        for i in range(0,500):
            self.updateError()
            self.updateWeights()
            

    def updateError(self):
        error=0
        for eachFile in self.fileData:
            token = eachFile["token"]
            value = 1
            for everyToken in token:
                value += token[everyToken]*self.weights[everyToken]
            eachFile["error"] = self.sigmoid(value)
            error+= eachFile["error"]
        

    def sigmoid(self,x):
        denom = 1+np.exp(-x)
        return (1/denom)
    
    def updateWeights(self):
        for token in self.weights.keys():
            val = 0
            error= 0
            for eachFile in self.fileData:
                tokens = eachFile["token"]
                trueValue = eachFile["class"]
                if token in tokens:
                    temp = trueValue-eachFile["error"]
                    error += temp
                    val +=tokens[token]*(temp)                
            #print (error)
            self.weights[token]+= ((val*self.rate)-(self.rate*self.lambdaVal*self.weights[token]))
       
    
    def test(self):
        
        hamFolder = os.listdir(self.testHam)
        hamCorrect = 0
        for each in hamFolder:
            hamDict = {}
            value = 0
            self.countWords(self.testHam+"/"+each,hamDict)
            for token in hamDict:
                if token in self.weights:
                    value+=self.weights[token]*hamDict[token]
            result =self.sigmoid(value)
            if result > 0.5:
                hamCorrect += 1
                
        accuracy = (float)(hamCorrect/len(hamFolder))*100   
        print("Ham accuracy is ",accuracy)
            
        spamFolder = os.listdir(self.testSpam)
        spamCorrect = 0
        for each in spamFolder:
            spamDict = {}
            
            value = 0
            self.countWords(self.testSpam+"/"+each,spamDict)
            
            for token in spamDict:
                if token in self.weights:
                    value+=self.weights[token]*spamDict[token]

            result =self.sigmoid(value)

            if result < 0.5:
                spamCorrect += 1
        
        accuracy = (float)(spamCorrect/len(spamFolder))*100   
        print("Spam accuracy is ",accuracy)
        
        totalAccuracy = (float)((spamCorrect+hamCorrect)/(len(hamFolder)+len(spamFolder)))*100
        print("Total accuracy is ",totalAccuracy)
        
