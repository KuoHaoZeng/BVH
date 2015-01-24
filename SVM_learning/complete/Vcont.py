import os, sys
import numpy as np

# Input list
class Vlist:
	videoPath = []
	videoName = []
	videoFolder = []
	videoLabel = []
	Len = []
	def __init__(self, inlist = None):
		if inlist != None and os.path.exists(inlist) == True:
			f = open(inlist, 'r')
			raw = f.readlines()
			f.close()

			for i in raw:
				self.videoPath.append(self.DigName(i, ' ', ' '))
				self.videoName.append(self.DigName(i, '/', '.'))
				self.videoFolder.append(self.DigName(i, ' ', '/' + self.DigName(i, '/', '.')) + '/')
				self.videoLabel.append(int(self.DigName(i, ' ', '\n')))
			self.Len = range(len(raw))
			print('---Total video number---')
			print(str(len(raw)) + '\n')
		else:
			raise ValueError("Wrong Input list")

	def DigName(self, Path, str1, str2):
		Temp = Path.split(str2)
		Temp = Temp[0].split(str1)
		return Temp[len(Temp) - 1]

# Control parameter
class Vcont_parameter:
	features_force = 0
	fisher_force = 0
	gmm_control = 0
	gmm_subsample = 0
	gmm_K = 16
	gmm_nit = 30
	gmm_redo = 1
	nthread = 1
	svm_C = 1
	svm_control = 0

	def __init__(self, options = None):
		if options == None:
			options = ''
		self.parse_options(options)

	def print_options(self, options):	
		print("---Parameters change---")
		string = ""
		i = 0
		while i < len(options):
			string += (options[i]+" ")
			i+=1
		if len(options) == 0:
			print('none\n')
		else:
			print(string + '\n')

	def parse_options(self, options):
		if isinstance(options, list):
			argv = options
		elif isinstance(options, str):
			argv = options.split()
		else:
			raise TypeError("arg 1 should be a list or a str.")

		i = 0
		while i < len(argv):
			if argv[i] == "-f":
				i += 1
				self.features_force = int(argv[i])
			elif argv[i] == "-v":
				i += 1
				self.fisher_force = int(argv[i])
			elif argv[i] == "-g":
				i += 1
				self.gmm_control = int(argv[i])
			elif argv[i] == "-s":
				i += 1
				self.gmm_subsample = int(argv[i])
			elif argv[i] == "-k":
				i += 1
				self.gmm_K = int(argv[i])
			elif argv[i] == "-ni":
				i += 1
				self.gmm_nih = int(argv[i])
			elif argv[i] == "-r":
				i += 1
				self.gmm_redo = int(argv[i])
			elif argv[i] == "-n":
				i += 1
				self.nthread = int(argv[i])
			elif argv[i] == "-c":
				i += 1
				self.svm_C = float(argv[i])
			elif argv[i] == "-m":
				i += 1
				self.svm_control = int(argv[i])
			else:
				raise ValueError("Wrong options")
			i += 1
		self.print_options(argv)

# Gmm model
class gmm_model:
	gmm = []
	mean = 0
	pca = np.empty(0, dtype = np.float32)

	def __init__(self, npz = None):
		self.gmm = [npz['w'], npz['mu'], npz['std']]
		self.pca = npz['pca']
		self.mean = npz['mean']
		print('---Gmm model has been generated---\n')
