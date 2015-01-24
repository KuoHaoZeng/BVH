import os
import sys
import subprocess

def Extracting(fulPath):
	subprocess.call('./Video ' + fulPath, shell = True)
	subprocess.call('./DenseTrackStab ' + fulPath, shell = True)

class Vlist:
	videoPath = []
	videoName = []
	def __init__(self, inlist = None):
		if inlist != None and os.path.exists(inlist) == True:
			f = open(inlist, 'r')
			raw = f.readlines()
			f.close()

			for i in raw:
				self.videoPath.append(i)
				self.videoName.append(self.DigName(i))

			print('---Total video number---')
			print(str(len(raw)) + '\n')
		else:
			raise ValueError("Wrong Input list")

	def DigName(self, Path):
		Temp = Path.split('.')
		Temp = Temp[0].split('/')
		return Temp[len(Temp)-1]

class Vcont_parameter:
	features_force = 0
	fisher_force = 0
	gmm_control = 0
	gmm_subsample = 0
	gmm_K = 16
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
				i = i + 1
				self.features_force = int(argv[i])
			elif argv[i] == "-v":
				i = i + 1
				self.fisher_force = int(argv[i])
			elif argv[i] == "-g":
				i = i + 1
				self.gmm_control = int(argv[i])
			elif argv[i] == "-s":
				i = i + 1
				self.gmm_subsample = int(argv[i])
			elif argv[i] == "-k":
				i = i + 1
				self.gmm_K = int(argv[i])
			elif argv[i] == "-c":
				i = i + 1
				self.svm_C = float(argv[i])
			elif argv[i] == "-m":
				i = i + 1
				self.svm_control = int(argv[i])
			else:
				raise ValueError("Wrong options")
			i += 1
		self.print_options(argv)
	
