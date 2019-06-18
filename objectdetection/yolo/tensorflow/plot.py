import matplotlib.pyplot as plt

with open('log', 'r') as f:
	loss = []
	ls = f.readlines()
	for l in fs:
		if 'train loss' in l:
			loss.append(l.split(': ')[-1])
	plt.plot(loss)
	plt.show()
