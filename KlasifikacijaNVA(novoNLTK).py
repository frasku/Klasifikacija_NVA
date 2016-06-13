# coding: utf-8
import sqlite3
from hmldb import HmlDB
import nltk
import math
import random
import re


db = HmlDB('hml.db')

def osobine(word):
    if len(word)==4:return{'First and last letter':word[0]+word[-1]}    
    if len(word)>4: return {'Last 3 letters':word[-3:]}
    if len(word)<=3: return {'Last 2 letters':word[-2:]}
    
    

imenice=[(token[0].lower(), 'I') for token in HmlDB.select_token_by_msd(db,'N%')]
pridjevi=[(token[0].lower(), 'P') for token in HmlDB.select_token_by_msd(db,'A%')]
glagoli=[(token[0].lower(), 'G') for token in HmlDB.select_token_by_msd(db,'V%')]

sve=set()
sve =imenice + pridjevi + glagoli
random.seed(4)# za jedinstveni shuffle
random.shuffle(sve)

lista_osobina = [(osobine(n), vrsta_rijeci) for(n, vrsta_rijeci) in sve]
tren_len=int(len(lista_osobina)*0.75)
trening, test= lista_osobina[:tren_len], lista_osobina[tren_len:]
classifier = nltk.NaiveBayesClassifier.train(trening)

###########TEST##############
x=int(len(sve)*0.75)
testni_print=sve[x+1:][:200]
print("+(pogodak)   -(promašaj)")
print("#"*60)
br=0
for rijec,vrsta in testni_print:
    if classifier.classify(osobine(rijec))==vrsta:print(rijec," ---> ",vrsta,"|classify: ",classifier.classify(osobine(rijec)),"|  +");br+=1
    else: print(rijec," ---> ",vrsta,"|classify: ",classifier.classify(osobine(rijec)),"|  -")

print("#"*60)
ukupno = len(testni_print)
preciznost = br / ukupno
print("preciznost nad 200 testnih riječi: ",preciznost)
#############################
#print(classifier.classify(osobine('trčim'))) #za unos bilo koje riječi

print("ukupna preciznost(25% testnih podataka(554199 riječi)):",nltk.classify.accuracy(classifier, test))
