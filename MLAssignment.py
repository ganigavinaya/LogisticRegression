#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 17:36:50 2017

@author: vinaya
"""
import sys

import LRwithStopWords
import LRwithoutStopWords
import NBwithStopWords
import NBwithoutStopWords

def main():
    if(len(sys.argv)<6):
        print("please input command line arguments: ")
        print("[training ham path] [training spam path] [test ham path] [test spam path] [stopwords]\n")
        return
    else:
        trainingHamPath = sys.argv[1]
        trainingSpamPath = sys.argv[2]
        testHamPath = sys.argv[3]
        testSpamPath = sys.argv[4]
        if(sys.argv[5]=="y" or (sys.argv[5]).lower()=="yes"):
            stopWords = "yes"
        else:
            stopWords =None
            
        if(stopWords !=None):
            print("------------------------------------------------")
            print("Logistic regression removing stop words")
            lr = LRwithStopWords.LogisticRegression(trainingHamPath,trainingSpamPath,testHamPath,testSpamPath)
            lr.run()
            lr.train()
            lr.test()
            
            print("------------------------------------------------")
            print("Naive Bayes removing stop words")
            nb = NBwithStopWords.NaiveBayes(trainingHamPath,trainingSpamPath,testHamPath,testSpamPath)
            nb.run()
            nb.train()
            nb.test()
        else:    
            print("------------------------------------------------")
            print("Logistic regression without removing stop words")
            lr = LRwithoutStopWords.LogisticRegression(trainingHamPath,trainingSpamPath,testHamPath,testSpamPath)
            lr.run()
            lr.train()
            lr.test()
            
            print("------------------------------------------------")
            print("Naive Bayes without removing stop words")
            nb = NBwithoutStopWords.NaiveBayes(trainingHamPath,trainingSpamPath,testHamPath,testSpamPath)
            nb.run()
            nb.train()
            nb.test()

main()        