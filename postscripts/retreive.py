import sys
import os
import re
from rouge import Rouge

def split_file(filename):

	with open(filename, "r") as f:

		f1 = open(filename + "_plabels","w")
		f2 = open(filename + "_tlabels", "w")
		pred_str=''
		true_str=''

		count = 0
		for lines in f:
		#	print count
			lines = lines.strip("\n")
			if (count % 2 == 0):
				#x = re.sub("<eos>", "", lines)
				x = lines.split("<eos>")[0]
				x = re.sub("Predicted summary ::", "",x)
				x = re.sub("<pad>", "",x)
				x = " ".join(x.split())

			else:
				x1 = lines.split("<eos>")[0]
				x1 = re.sub("True labels ::","",x1)
				x1 = re.sub("<pad>", "",x1)
				x1 = " ".join(x1.split())
				if not(x1 is " " or x1 is ""):
					f1.write(x + " <eos>\n")
					f2.write(x1 + " <eos>\n")
					pred_str+=x
					true_str+=x1
			count = count + 1
		robj=Rouge()
		scores = robj.get_scores(true_str,pred_str)[0]
		
		for key in sorted(scores.keys()):
			print(key)
			for s in scores[key].keys():
				if s=='r':
					print(s+":"+str(scores[key][s]*100))

		

def main():

	split_file(sys.argv[1])


if __name__ == '__main__':
	main()
