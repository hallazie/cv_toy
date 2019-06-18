import tensorflow as tf

if __name__ == '__main__':
	a = tf.constant([[[1,2], [3,4]], [[1,2], [3,4]]])
	b = a[:,:,:1]
	# b = a[:,tf.newaxis,:]
	# c = a[tf.newaxis,:]
	# s = tf.Session()
	# print(s.run(a))
	# print(s.run(b))
	# print(s.run(c))
	# print(c.shape)
	print(b.shape)