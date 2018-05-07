#!/usr/bin/python

from __future__ import division

from __future__ import with_statement

import os
import numpy as np
from scipy.integrate import simps
import itertools
import glob
import sys
from numba import jit
from sys import stderr

'''
Read a group of scoring function weights and then generate docking results for each weights combination
Output the total hits, hit number on top X for each case and each weights combination
Usage: python ./opti_it0.py <.score files directory> <weight file>
'''

def print_err(*args, **kwargs):
	sys.stderr.write(' '.join(map(str,args)) + '\n')
	

def read_weights(file):

	'''Read the range file and extract the ranges of weights'''

	with open(file) as inp:
		weights = np.loadtxt(inp)

	Evdw = list(weights[:,0])
	Edesolv = list(weights[:,1])
	Eair = list(weights[:,2])
	BSA = list(weights[:,3])
	'''
	for i in range(np.size(weights, axis = 0)):
		Evdw.append(weights[i,0])
		Edesolv.append(weights[i,1])
		Eair.append(weights[i,2])
		BSA.append(weights[i,3])
	'''
	
	return Evdw, Edesolv, Eair, BSA


def parse_scores(scores_file):

	'''Parses the scores file (X.scores). It DOES NOT parse the identifier (csv)'''

	with open(scores_file) as inp:
		matrix = np.loadtxt(inp, usecols=(1, 2, 3, 4, 5, 6, 7))
    
	return matrix
	

@jit(nopython=True)
def calculate_recall(irmsd, topN):

	'''Calculates the recall for a given top (topN) for a list of i-RMSD'''
	
	criteria = 4.0
	N_acceptable = 0
	T_acceptable = 0
	
	for r in list(irmsd):
		if r <= criteria:
			T_acceptable += 1
		else:
			continue
	
	for r in list(irmsd[0:topN]):
		if r <= criteria:
			N_acceptable += 1
		else:
			continue
	
	if N_acceptable == topN or N_acceptable == T_acceptable:
		recall = 1.0
	elif T_acceptable != topN:
		recall = N_acceptable / min(topN,T_acceptable)
	else:
		recall = 0.0
	
	return recall


@jit(nopython=True)
def cal_std_auc(irmsd):

	'''Calculates standard AUC (based on the true positive number in top 50, 100, 200)
	'''

	cut_5 = cut_10 = cut_25 = cut_50 = cut_100 = cut_200 = 0
	criteria = 2.0

	for n, r in enumerate(list(irmsd[:200]), 1):
		if r <= criteria:
			if n <= 5:
				cut_5 += 1
			elif n <= 10:
				cut_10 += 1
			elif n <= 25:
				cut_25 += 1
			elif n <= 50:
				cut_50 += 1
			elif n <= 100:
				cut_100 += 1
			else:
				cut_200 += 1

	cut_10  += cut_5
	cut_25  += cut_10
	cut_50  += cut_25
	cut_100 += cut_50
	cut_200 += cut_100
		
	max_5   = cal_max_acc(irmsd, 5)
	max_10  = cal_max_acc(irmsd, 10)
	max_25  = cal_max_acc(irmsd, 25)
	max_50  = cal_max_acc(irmsd, 50)
	max_100 = cal_max_acc(irmsd, 100)
	max_200 = cal_max_acc(irmsd, 200)

	std_auc = (5 * cut_5 / max_5 + 10 * cut_10 / max_10 + 20 * cut_25 / max_25 + 37.5 * cut_50 / max_50 + 75 * cut_100 / max_100 + \
		50 * cut_200 / max_200) * 1.0 / 197.5
	
	return std_auc

@jit(nopython=True)
def cal_max_acc(irmsd, topN):

	'''Calculates the maximum complex number where complexes are hits in the topN (HS ranking, whose irmsd <= 2.0)
	'''

	max_acc = 0
	for n, item in enumerate(irmsd, 1):
		if n >= topN: break
		if item <= 2.0: max_acc += 1

	return max_acc
		

#@jit(nopython=True)
def count_hits(energy_terms, irmsd, combi):

	'''MAIN FUNCTION. You may optimize for recall, AUC or min number of good models. NOTE: only 4 terms taken
		into consideration'''

	std_auc = []
	
	hscore = (energy_terms[:,0] + energy_terms[:,1] * combi[0] + energy_terms[:,2] * combi[1] + energy_terms[:,3] * combi[2] + \
		energy_terms[:,4] * combi[3])
	new = np.column_stack((irmsd, hscore))
	sorted_new = new[np.argsort(new[:, 1])]
	sorted_irmsd = sorted_new[:, 0]
	
	hits = [cal_max_acc(sorted_irmsd, 5), cal_max_acc(sorted_irmsd, 10), cal_max_acc(sorted_irmsd, 25), cal_max_acc(sorted_irmsd, 50), \
		cal_max_acc(sorted_irmsd, 100), cal_max_acc(sorted_irmsd, 200)]
		
		#i = 1
		#total = recall[0] + recall[-1]
		
		#for r in recall[1:-1]:
		#	if i%2 == 0:
		#		total += 2*r
		#	else:
		#		total += 4*r
		#	i += 1
		#auc[n] = total * (1/3.0)
	
	return hits

if __name__ == '__main__':

	Evdw, Edesolv, Eair, BSA = read_weights(sys.argv[2]) #Reading the ranges from range file

	combinations = np.array([Evdw, Edesolv, Eair, BSA])
	
	file_list = [each for each in os.listdir(sys.argv[1]) if each.endswith('.scores')]
	invalid_list = []

	all_hits = []
	all_level_hits = []
	
	for n in range(np.size(combinations, axis = 1)):
		for file in file_list:
			matrix = parse_scores(sys.argv[1] + file)

			if min(matrix[:,0]) > 2.0:
				invalid_list.append(file)
				continue

			energy_terms = matrix[:, [2,3,4,5,6]]
			
			all_hits.append(cal_max_acc(matrix[:,0], matrix.shape[0]))
			level_hits = count_hits(energy_terms, matrix[:,0], [combinations[0,n], combinations[1,n], combinations[2,n], combinations[3,n]])
			
			#One docking case with all weights combinations
			all_level_hits.append(level_hits)
	
	with open("hits_ana.txt", 'w') as out:
		out.write("#Case name\tAll hits\tTop5 hits\tTop10 hits\tTop25 hits\tTop50 hits\tTop100 hits\tTop200 hits\tWeights combinations\n")
		i = 0
		for n in range(np.size(combinations, axis = 1)):
			for file in file_list:
				if file in invalid_list:
					continue
				out.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(file[0:8], all_hits[i], all_level_hits[i][0], \
					all_level_hits[i][1], all_level_hits[i][2], all_level_hits[i][3], all_level_hits[i][4], all_level_hits[i][5], \
						combinations[0,n], combinations[1,n], combinations[2,n], combinations[3,n]))
				i += 1

			
			
		




	
		
		
		
		
		

		
		
		
		