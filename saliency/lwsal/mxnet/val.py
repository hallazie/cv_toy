def it(n):
	for i in range(n):
		yield i**2

iterator = it(10)
for i in range(20):
	try:
		p = next(iterator)
		print(p)
		print(type(p))
	except StopIteration:
		print('just stoped')