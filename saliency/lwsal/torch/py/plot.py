import matplotlib.pyplot as plt

def plot_loss():
	losses = []
	with open('log', 'r') as log:
		lines = log.readlines()
		for line in lines:
			loss = line.split('MAELoss = ')[-1]
			losses.append(float(loss))
		plt.plot(losses)
		plt.show()

if __name__ == '__main__':
	plot_loss()