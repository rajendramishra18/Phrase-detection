import cPickle as cp
import numpy as np
import csv

# Glabal variables
NUM_WORDS = 30

def format_result(word_file , pred_file , result_file):
	# Load the word list
	word_list = cp.load(open(word_file,'r'))
	
	# Load the prediction file
	pred = cp.load(open(pred_file , 'r'))
	
	# Open a csv file to write the result
	fp = open(result_file , 'wb')
	csv_writer = csv.writer(fp, delimiter='\t')
	
	for i in range(0 , len(word_list)):
		each = word_list[i]
		result = pred[i]
		phrase_list = []
		for j in range(0,len(each)):
			
			# index shall not exceed 30 and their are two classes in result.
			# The class with greater value will decide whether to include the word in phrase list or not.
			if j<NUM_WORDS and result[j][0] < result[j][1]:
				phrase_list.append(each[j])
				
				
		phrase_string = ' '.join(phrase_list[0:len(phrase_list)])
		sent_string = ' '.join(each[0:len(each)])
		csv_writer.writerow([sent_string.encode('utf-8') , phrase_string.encode('utf-8')])
		
	
if __name__ == "__main__":
	format_result("word_list_2.cp" , "pred_2.cp" , "results_2.csv")
