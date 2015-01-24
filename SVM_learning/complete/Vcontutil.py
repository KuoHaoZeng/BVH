from Vcont import *
import sys

print

Vlist(sys.argv[1])

options = []
i = 2
while i < len(sys.argv):
	options.append(sys.argv[i])
	i += 1

Vcont_parameter(options)
