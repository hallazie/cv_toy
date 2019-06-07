import mxnet as mx

def conv_fl(data, num_filter, kernel=(3,3), stride=(1,1), pad=(1,1)):
	conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
	norm = mx.symbol.BatchNorm(data=conv)
	relu = mx.symbol.LeakyReLU(data=norm)
	return relu

def conv_dw(data, oup, exp):
	conv1 = mx.symbol.Convolution(data=data, num_filter=exp, kernel=(1,1), stride=(1,1), pad=(0,0), num_group=1)
	norm1 = mx.symbol.BatchNorm(data=conv1)
	relu1 = mx.symbol.LeakyReLU(data=norm1)
	conv2 = mx.symbol.Convolution(data=relu1, num_filter=exp, kernel=(3,3), stride=(1,1), pad=(1,1), num_group=exp)
	norm2 = mx.symbol.BatchNorm(data=conv2)
	relu2 = mx.symbol.LeakyReLU(data=norm2)
	conv3 = mx.symbol.Convolution(data=relu2, num_filter=oup, kernel=(1,1), stride=(1,1), pad=(0,0), num_group=1)
	norm3 = mx.symbol.BatchNorm(data= conv3)
	return norm3

def pool(data):
	return mx.symbol.Pooling(data=data, pool_type='max', kernel=(2,2), stride=(2,2))

def subnet(data):
	c1 = conv_fl(data, 32)
	p1 = pool(c1)
	c2 = conv_dw(p1, 16, 64)
	c3 = conv_dw(c2, 16, 64)
	p3 = pool(c3)
	c4 = conv_dw(p3, 24, 96)
	c5 = conv_dw(c4, 24, 96)
	c6 = conv_dw(c5, 24, 96)
	p6 = pool(c6)
	c7 = conv_dw(p6, 32, 128)
	c8 = conv_dw(c7, 32, 128)
	c9 = conv_dw(c8, 32, 128)
	c10 = conv_dw(c9, 32, 128)
	c11 = conv_dw(c10, 64, 256)
	c12 = conv_dw(c11, 64, 256)
	c13 = conv_dw(c12, 128, 512)
	c14 = conv_fl(c13, 1, kernel=(1,1), stride=(1,1), pad=(0,0))
	return c14

def overfit_subnet(data):
	c1 = conv_fl(data, 32)
	c2 = conv_fl(c1, 32)
	p2 = pool(c2)
	c3 = conv_fl(p2, 64)
	c4 = conv_fl(c3, 64)
	p4 = pool(c4)
	c5 = conv_fl(p4, 128)
	c6 = conv_fl(c5, 128)
	c7 = conv_fl(c6, 128)
	p7 = pool(c7)
	c8 = conv_fl(p7, 256)
	c9 = conv_fl(c8, 256)
	c10 = conv_fl(c9, 256)
	c11 = conv_fl(c10, 1, kernel=(1,1), stride=(1,1), pad=(0,0))
	return c11

def net():
	label = mx.symbol.Variable('label')
	data_coarse = mx.symbol.Variable(name='coarse')
	data_fine = mx.symbol.Variable(name='fine')
	out_coarse = subnet(data_coarse)
	out_fine = subnet(data_fine)
	out = out_coarse * out_fine
	return mx.symbol.MAERegressionOutput(data=out, label=label)

def overfit():
	label = mx.symbol.Variable('label')
	data_coarse = mx.symbol.Variable(name='coarse')
	data_fine = mx.symbol.Variable(name='fine')
	out_coarse = overfit_subnet(data_coarse)
	out_fine = overfit_subnet(data_fine)
	out = out_coarse * out_fine
	return mx.symbol.LinearRegressionOutput(data=out, label=label)	