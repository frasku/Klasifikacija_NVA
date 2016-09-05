# coding: utf-8
import sqlite3
from hmldb import HmlDB
import nltk
import math
import random
import re
import timeit


db = HmlDB('hml.db')


def osobine(word): #osobine1
    if len(word)>10:return{'First and last letter':word[-6:]}
    if len(word)==10:return{'First and last letter':word[-5:]}
    if len(word)==5:return{'First and last letter':word[-3:]}
    if len(word)==4:return{'First and last letter':word[0]+word[-1]}     
    if len(word)>5 and len(word)<10: return {'Last 3 letters':word[-4:]}
    if len(word)<=3: return {'Last 2 letters':word[-2:]}
##def osobine(word): #osobine2
##    if len(word)==4:return{'First and last letter':word[0]+word[-1]}    
##    if len(word)>4: return {'Last 3 letters':word[-3:]}
##    if len(word)<=3: return {'Last 2 letters':word[-2:]}
##def osobine(word):#osobine 3
##    return {'Last 3 letters':word[-3:]}      

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
print("TESTTTTT",len(test))
start1 = timeit.default_timer()
classifierNB = nltk.NaiveBayesClassifier.train(trening)
stop1 = timeit.default_timer()

#############TEST##############
x=int(len(sve)*0.75)
testni_print=sve[x+1:][:200]
print("+(pogodak)   -(promašaj)")
print("#"*60)
im,pr,gl,nijeIm,nijePr,nijeGl=0,0,0,0,0,0
for rijec,vrsta in testni_print:
    if vrsta=='I':
        if classifierNB.classify(osobine(rijec))==vrsta:print(rijec," ---> ",vrsta,"|classify: ",classifierNB.classify(osobine(rijec)),"|  +");im+=1
        else:print("-"*60,"Imenica, model kaže ",classifierNB.classify(osobine(rijec)),"|  -");nijeIm+=1
    if vrsta=='P':
        if classifierNB.classify(osobine(rijec))==vrsta:print(rijec," ---> ",vrsta,"|classify: ",classifierNB.classify(osobine(rijec)),"|  +");pr+=1
        else: print("-"*60,"Pridjev, model kaže ",classifierNB.classify(osobine(rijec)),"|  -");nijePr+=1
    if vrsta=='G':
        if classifierNB.classify(osobine(rijec))==vrsta:print(rijec," ---> ",vrsta,"|classify: ",classifierNB.classify(osobine(rijec)),"|  +");gl+=1
        else: print("-"*60,"Glagol, model kaže ",classifierNB.classify(osobine(rijec)),"|  -");nijeGl+=1

print("#"*60)
ukupno = len(testni_print)
preciznost = (im+pr+gl) / ukupno
print("Preciznost nad 200 testnih riječi: ",round(preciznost,3),(im+pr+gl))
print("Im, NijeIm,Pr, NijePr,Gl, NijeGl -----> ",im,nijeIm,pr,nijePr,gl,nijeGl)
#############################

print("Ukupna preciznost NBayes(25% testnih podataka(554199 riječi)):",nltk.classify.accuracy(classifierNB, test))
print("Vrijeme NBayes: ",round(stop1-start1,3)," sec")

################### Maximum Entropy Classifier #########################
start = timeit.default_timer()
classifierMaxE=nltk.MaxentClassifier.train(trening)
stop = timeit.default_timer()

print("+(pogodak)   -(promašaj)")
print("#"*60)
im,pr,gl,nijeIm,nijePr,nijeGl=0,0,0,0,0,0

for rijec,vrsta in testni_print:
    if vrsta=='I':
        if classifierMaxE.classify(osobine(rijec))==vrsta:print(rijec," ---> ",vrsta,"|classify: ",classifierMaxE.classify(osobine(rijec)),"|  +");im+=1
        else:print("-"*60,"Imenica, model kaže ",classifierMaxE.classify(osobine(rijec)),"|  -");nijeIm+=1
    if vrsta=='P':
        if classifierMaxE.classify(osobine(rijec))==vrsta:print(rijec," ---> ",vrsta,"|classify: ",classifierMaxE.classify(osobine(rijec)),"|  +");pr+=1
        else: print("-"*60,"Pridjev, model kaže ",classifierMaxE.classify(osobine(rijec)),"|  -");nijePr+=1
    if vrsta=='G':
        if classifierMaxE.classify(osobine(rijec))==vrsta:print(rijec," ---> ",vrsta,"|classify: ",classifierMaxE.classify(osobine(rijec)),"|  +");gl+=1
        else: print("-"*60,"Glagol, model kaže ",classifierMaxE.classify(osobine(rijec)),"|  -");nijeGl+=1

print("#"*60)
ukupno = len(testni_print)
preciznost = (im+pr+gl) / ukupno
print("Preciznost nad 200 testnih riječi: ",round(preciznost,3),(im+pr+gl))
print("Im, NijeIm,Pr, NijePr,Gl, NijeGl -----> ",im,nijeIm,pr,nijePr,gl,nijeGl)
#############################


print("Ukupna preciznost MaxEntropy:",nltk.classify.accuracy(classifierMaxE, test))
print ("Vrijeme MaxEntropy: ",round((stop - start)/60,3)," min")

