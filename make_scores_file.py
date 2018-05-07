#!/usr/bin/python

import sys
import os
import re
import numpy as np
import csv

'''Given the PDBs and the i-RMSD.dat, this script generates the .scores file'''

complex = sys.argv[1]
path = '/data/zyzhang/work/had_lig/' + complex + '/run1/structures/it0/'

#Eelec, Evdw, Edesolv, Eair, BSA
weights = [1.0, 1.0, 1.0, 0.01, -0.01]

def get_rmsd(ranking):

	with open(ranking) as inp:
		next(inp)
		rmsd = [lines.strip().split()[2] for lines in inp]
		
	return rmsd

def get_haddock_terms(file_nam, ranking):

	with open(ranking) as inp:
		next(inp)
		rmsd = [lines.strip().split()[1] for lines in inp]
	
	with open(file_nam,'rb') as inp:
	#	next(inp)
		os.chdir(path)
		haddock_terms = []
		for file, rms in zip(inp, rmsd):
			structure = file.strip().split()[0]
			print structure
			try:
				with open(structure, 'rb') as pdb:
					for line in pdb:
						if line[0:6] == 'REMARK':
							if re.match(r'REMARK energies', line):
								ene = line.split(',')
								Eelec = ene[6].strip()
								Evdw = ene[5].strip()
								Eair = ene[7].strip()
							elif re.match(r'REMARK Desolvation energy', line):
								Edesolv = line.split()[3].strip()
							elif re.match(r'REMARK buried surface area', line):
								BSA = line.split()[4].strip()
							else:
								continue
					energy = map(float,[Eelec, Evdw, Edesolv, Eair, BSA])
					hs = np.dot(energy, weights)
					terms = [structure, rms, Eelec, Evdw, Edesolv, Eair, BSA, str(hs)]
					haddock_terms.append(terms)
			except:
				pass
        
	return haddock_terms

if __name__ == '__main__':

	print "Extracting haddock terms"
	
	haddock_terms = get_haddock_terms(path + 'file.nam', path + 'i-RMSD.dat')
        matrix = sorted(haddock_terms, key=lambda tup: tup[7], reverse=True)
	
	print "Creating file"
	
	with open('/data/zyzhang/work/lpg_ana/scores/' + complex + '.scores', 'wb') as outp:
		writer = csv.writer(outp, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
		for row in matrix:
			writer.writerow(row)
	
	
	
