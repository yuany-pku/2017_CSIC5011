# -*- coding: utf-8 -*-
"""
Created on Fri Dec 08 17:27:31 2017

@author: Thinkpad
"""
from __future__ import division

from pandas import read_excel
import numpy as np
from hmm_class import HMM, read_hmm, read_sequence
from pandas import read_excel
import math
import sys



data = read_excel('data.xlsx')

left_photo = data['photo_id_left']
right_photo = data['photo_id_right']
choice = data['choice']

score = read_excel('score.xlsx')

index_photo = score['Method']
specific_score = score['Arcsin']

photo_choice_map = {}

for i in range(len(index_photo)):
    
    photo_choice_map[index_photo[i]] = specific_score[i]
    
    
def compare(a,b,c):
    
    if(c==1):
        
        if(a>b): return 'right'
        else: return 'wrong'
        
    if(c==-1):
        
        if(a>b): return 'wrong'
        else: return 'right'
    

choice_1 = 0
one_right = 0
one_wrong = 0
choice_minus_1 = 0
minus_one_right = 0
minus_one_wrong = 0

choices = []

a = 0
    

    
    
for j in range(len(choice)):
   
    if(choice[j]== 1):

       
        choice_1 += 1
        if(compare(photo_choice_map.get(left_photo[j]),photo_choice_map.get(\
                   right_photo[j]),1) == 'right'):
            
            one_right += 1
            
        else: one_wrong += 1
        
    elif(choice[j]== -1):
        
        choice_minus_1 += 1
        if(compare(photo_choice_map.get(left_photo[j]),photo_choice_map.get(\
                   right_photo[j]),-1) == 'right'):   
            
            minus_one_right += 1
        else: minus_one_wrong += 1
        
 
givenOneRight = one_right/choice_1
givenOneWrong = 1 - givenOneRight  

givenMinusOneRight = minus_one_right/choice_minus_1
givenMinusOneWrong = 1 - givenMinusOneRight






f = open('1.txt','w')       
f_f = open('11.txt','w')  
f_f.write('T= ' + str(len(choice))+'\n')

choices = []
zz =[]

for z in range(len(choice)):
    
    if(choice[z]==0): continue
    choices.append(str(choice[z]))
    if(compare(photo_choice_map.get(left_photo[z]),photo_choice_map.get\
               (right_photo[z]),choice[z]) == 'right'):
        zz.append('right')
        
    else: zz.append('wrong')
    


yone_one = 0
yone_minus_one = 0
yminus_one_minus_one = 0
yminus_one_one = 0

for h in range(len(choices)-1):
    
    if(choices[h] == '1'):
        if(choices[h+1] == '1'): yone_one += 1
        elif(choices[h+1] == '-1'):  yone_minus_one += 1   
        
    elif(choices[h] == '-1'):
        if(choices[h+1] == '1'): yminus_one_one += 1
        elif(choices[h+1] == '-1'): yminus_one_minus_one += 1
     
            

    
    
    
   
for k in range(len(choices)):
    
    f.write( choices[k] +',' + zz[k] + '\n' )
            
    
for g in range(len(choices)-1):
    
    f_f.write(choices[g] + ',')

f_f.write(choices[len(choices)-1])    
f.close()  
f_f.close()   
        
        
hmmfile = 'initial_parameters.txt'
seqfile = '11.txt'

M, N, pi, A, B =read_hmm(hmmfile)
T, obs = read_sequence(seqfile)

hmm_object = HMM(pi, A, B)
logprobinit, logprobfinal, a, b = hmm_object.baum_welch(obs)    
    
        
StateMap = {'1' : 0, '-1' : 1}
StateIndex = {0 : '1', 1 : '-1'}

ObsMap = {'right' : 0, 'wrong' : 1}
             
ObsIndex = {0 : 'right', 1 : 'wrong'}
               

Prob = [0.5, 0.5]


c = np.transpose(b)
TProb = a
EProb = c    

    
# Using the prior probabilities and state map, return:
#     P(state)
def getStatePriorProb(prob, stateMap, state):
   return prob[stateMap[state]]

# Using the transition probabilities and state map, return:
#     P(next state | current state)
def getNextStateProb(tprob, stateMap, current, next):
   return tprob[stateMap[current]][stateMap[next]]

# Using the observation probabilities, state map, and observation map, return:
#     P(observation | state)
def getObservationProb(eprob, stateMap, obsMap, state, obs):
   return eprob[obsMap[obs]][stateMap[state]]

# Normalize a probability distribution
def normalize(pdist):
   s = sum(pdist)
   for i in range(0,len(pdist)):
      pdist[i] = pdist[i] / s
   return pdist


# Filtering.
# Input:  The HMM (state and observation maps, and probabilities) 
#         A list of T observations: E(0), E(1), ..., E(T-1)
#         (ie whether the umbrella was seen [yes, no, ...])
#
# Output: The posterior probability distribution over the most recent state
#         given all of the observations: P(X(T-1)|E(0), ..., E(T-1)).
def filter( \
   stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, observations):
   # intialize probability distribution to prior
   pdist = prob
   # update for subsequent times 
   for k in range(0,len(observations)):
      pdist_new = []
      for i in range(0,len(stateMap)):
         prob_i = 0.0
         for j in range(0,len(stateMap)):
            prob_i = prob_i + tprob[j][i] * pdist[j]
         prob_i = prob_i * eprob[obsMap[observations[k]]][i]
         pdist_new.append(prob_i)
      pdist = normalize(pdist_new)
   return pdist


# Prediction.
# Input:  The HMM (state and observation maps, and probabilities) 
#         A list of T observations: E(0), E(1), ..., E(T-1)
#
# Output: The posterior probability distribution over the next state
#         given all of the observations: P(X(T)|E(0), ..., E(T-1)).
def predict( \
   stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, observations):
   # compute P(X(T-1)|E(0), ..., E(T-1))
   pdist = filter( \
      stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, observations)
   # predict next timestep
   pdist_next = []
   for i in range(0,len(stateMap)):
      prob_i = 0.0
      for j in range(0,len(stateMap)):
         prob_i = prob_i + tprob[j][i] * pdist[j]
      pdist_next.append(prob_i)
   return pdist_next


# Smoothing.
# Input:  The HMM (state and observation maps, and probabilities) 
#         A list of T observations: E(0), E(1), ..., E(T-1)
#
# Ouptut: The posterior probability distribution over each state given all
#         of the observations: P(X(k)|E(0), ..., E(T-1) for 0 <= k <= T-1.
#
#         These distributions should be returned as a list of lists. 
def smooth( \
   stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, observations):
   # compute forward messages
   fv = prob
   forward = []
   for k in range(0,len(observations)):
      fv_next = []
      for i in range(0,len(stateMap)):
         prob_i = 0.0
         for j in range(0,len(stateMap)):
            prob_i = prob_i + tprob[j][i] * fv[j]
         prob_i = prob_i * eprob[obsMap[observations[k]]][i]
         fv_next.append(prob_i)
      fv = normalize(fv_next)
      forward.append(fv)
   # compute backward messages
   sv = []
   b = []
   for i in range(0,len(stateMap)):
      b.append(1)
   for k in range(len(observations)-1,-1,-1):
      sv_curr = []
      for i in range(0,len(stateMap)):
         sv_curr.append(forward[k][i]*b[i])
      sv.append(normalize(sv_curr))
      b_prev = []
      for i in range(0,len(stateMap)):
         prob_i = 0.0
         for j in range(0,len(stateMap)):
            prob_i = \
               prob_i + eprob[obsMap[observations[k]]][j] * b[j] * tprob[i][j]
         b_prev.append(prob_i)
      b = normalize(b_prev)
   sv.reverse()
   return sv


# Viterbi algorithm.
# Input:  The HMM (state and observation maps, and probabilities) 
#         A list of T observations: E(0), E(1), ..., E(T-1)
#      
# Output: A list containing the most likely sequence of states.
#         (ie [sunny, foggy, rainy, sunny, ...])
def viterbi( \
   stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, observations):
   # intialize probability distribution to prior
   pdist = prob
   # update max probability paths, keeping back links to best path
   back_pointers = []
   for k in range(0,len(observations)):
      pdist_new = []
      prev_pointer = []
      for i in range(0,len(stateMap)):
         prob_i = 0.0
         best_j = 0
         for j in range(0,len(stateMap)):
            prob_i_from_j = tprob[j][i] * pdist[j]
            if (prob_i_from_j > prob_i):
               prob_i = prob_i_from_j
               best_j = j
         prob_i = prob_i * eprob[obsMap[observations[k]]][i]
         pdist_new.append(prob_i)
         prev_pointer.append(best_j)
      pdist = normalize(pdist_new)
      back_pointers.append(prev_pointer)
   # extract the best path
   n = len(observations) - 1
   s_prob = 0
   s = 0
   for i in range(0,len(stateMap)):
      if (pdist[i] > s_prob):
         s_prob = pdist[i]
         s = i
   seq = []
   for k in range(n,-1,-1):
      seq.append(stateIndex[s])
      s = back_pointers[k][s]
   seq.reverse()
   return seq


# Functions for testing.
# You should not change any of these functions.
def loadData(filename):
   input = open(filename, 'r')
   
   data = []
   for i in input.readlines():
      x = i.split()
      y = x[0].split(",")
      data.append(y)
   return data

def accuracy(a,b):
   total = float(max(len(a),len(b)))
   c = 0
   for i in range(min(len(a),len(b))):
      if a[i] == b[i]:
         c = c + 1          
   return c/total

def test(stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, data):
   observations = []
   classes = []
   for c,o in data:
      observations.append(o)
      classes.append(c)
   n_obs_short = len(observations)
   obs_short = observations[0:n_obs_short]
   classes_short = classes[0:n_obs_short]
   print 'Short observation sequence:'
   print '   ', obs_short
   # test filtering 
   result_filter = filter( \
      stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, obs_short)
   print '\nFiltering - distribution over most recent state:'
   for i in range(0,len(result_filter)):
      print '   ', stateIndex[i], '%1.3f' % result_filter[i],
   # test prediction
   result_predict = predict( \
      stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, obs_short)
   print '\n\nPrediction - distribution over next state:'
   for i in range(0,len(result_filter)):
      print '   ', stateIndex[i], '%1.3f' % result_predict[i],
   # test smoothing
   result_smooth = smooth( \
      stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, obs_short)
   print '\n\nSmoothing - distribution over state at each point in time:'
   for t in range(0,len(result_smooth)):
      result_t = result_smooth[t]
      print '   ', 'time', t,
      for i in range(0,len(result_t)):
         print '   ', stateIndex[i], '%1.3f' % result_t[i],
      print ' '
   # test viterbi
   result_viterbi = viterbi( \
      stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, obs_short)
   print '\nViterbi - predicted state sequence:\n   ', result_viterbi
   print 'Viterbi - actual state sequence:\n   ', classes_short 
   print 'The accuracy of your viterbi classifier on the short data set is', \
      accuracy(classes_short, result_viterbi)
   result_viterbi_full = viterbi( \
      stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, observations)
   print 'The accuracy of your viterbi classifier on the entire data set is', \
      accuracy(classes, result_viterbi_full)
   print len(data)   
      


data = loadData('1.txt')

our_test = test(StateMap, StateIndex, ObsMap, ObsIndex, \
     Prob, TProb, EProb, data)

print a
print c    


print givenOneRight , givenOneWrong 
print givenMinusOneRight ,givenMinusOneWrong    
    
    
print(yone_one/(yone_one+yone_minus_one))
print(yone_minus_one/(yone_one+yone_minus_one))            
print(yminus_one_one/(yminus_one_one+yminus_one_minus_one))   
print(yminus_one_minus_one/(yminus_one_one+yminus_one_minus_one))      
        
    
    
    