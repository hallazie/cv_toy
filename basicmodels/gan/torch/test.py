# coding:utf-8
#
# @author:xsh

import pandas as pd

def calc_acc_main():
	file_path = u'data/stemi_ICD10编码_前50.csv'
	df = pd.read_csv(file_path).values
	for i, row in enumerate(df):
		name = [row[i] for i in range(row) if i%2==0]
		print(repr(name).decode('unicode-escape'))


def calc_recall():
	pass


if __name__ == '__main__':
	calc_acc_main()