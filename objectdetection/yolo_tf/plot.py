import matplotlib.pyplot as plt

with open('log', 'r') as f:
	loss = []
	ls = f.readlines()
	for l in ls:
		if 'train loss' in l:
			try:
				print(float(l.split(': ')[-1].replace('\n', '')))
				loss.append(float(l.split(': ')[-1].replace('\n', '')))
			except Exception as e:
				print(e)
	plt.plot(loss)
	plt.show()
