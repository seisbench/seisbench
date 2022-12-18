import torch
import torch.nn as nn

class first_unpooled_conv_then_pooled_conv(nn.Module):
	def __init__(self, in_filters=1, out_filters=2, kernel=2, pool=1):
		super().__init__()
		# no pooling
		self.unpooled = nn.Conv1d(in_filters, out_filters, kernel,
				padding='same')
		# pooling
		self.pooled = nn.Conv1d(out_filters, out_filters, kernel,
				pool)
		self.bn = nn.BatchNorm1d(out_filters)
		self.activation = torch.relu
	def forward(self, x):
		leftskip = self.activation(self.bn(self.unpooled(x)))
		out = self.activation(self.bn(self.pooled(leftskip)))
		return leftskip, out

class conv_then_deconv_then_concat(nn.Module):
	def __init__(self, in_filters=1, out_filters=2, kernel=7, pool=4, padding=0):
		super().__init__()
		# conv
		self.conv = nn.Conv1d(in_filters, out_filters, kernel, padding='same')
		# deconv
		self.deconv = nn.ConvTranspose1d(out_filters, out_filters,
				kernel, pool, output_padding=padding)
		self.bn = nn.BatchNorm1d(out_filters)
		self.activation = torch.relu
	def forward(self, leftskip,x):
		conved = self.activation(self.bn(self.conv(x)))
		deconved = self.activation(self.bn(self.deconv(conved)))
		return torch.cat([leftskip, deconved], dim=1)

class torch_version_PhaseNet(nn.Module):
	def __init__(
		self, in_channels=3, classes=3, phases="NPS", sampling_rate=100, **kwargs
		):
		super().__init__()
		self.in_channels = in_channels
		self.classes = classes
		self.kernel_size = 7
		self.stride = 4
		self.filters_root = 8
		self.activation = torch.relu
		self.depths = 4
		self.layernum = [2**x*self.filters_root for x in range(self.depths)]

		self.inc = nn.Conv1d(self.in_channels, self.filters_root,
				self.kernel_size, padding='same')
		self.in_bn = nn.BatchNorm1d(self.filters_root)
		# downsampling convolution
		self.dconv1 = first_unpooled_conv_then_pooled_conv(
				self.filters_root,
				self.layernum[0],
				self.kernel_size,
				self.stride)
		self.dconv2 = first_unpooled_conv_then_pooled_conv(
				self.layernum[0],
				self.layernum[1],
				self.kernel_size,
				self.stride)
		self.dconv3 = first_unpooled_conv_then_pooled_conv(
				self.layernum[1],
				self.layernum[2],
				self.kernel_size,
				self.stride)
		self.dconv4 = first_unpooled_conv_then_pooled_conv(
				self.layernum[2],
				self.layernum[3],
				self.kernel_size,
				self.stride)
		# upsampling convolution
		self.uconv1 = conv_then_deconv_then_concat(
				self.layernum[3],
				self.layernum[3],
				self.kernel_size,
				self.stride,
				2)
		self.uconv2 = conv_then_deconv_then_concat(
				self.layernum[3]*2,
				self.layernum[2],
				self.kernel_size,
				self.stride,
				3)
		self.uconv3 = conv_then_deconv_then_concat(
				self.layernum[2]*2,
				self.layernum[1],
				self.kernel_size,
				self.stride,
				2)
		self.uconv4 = conv_then_deconv_then_concat(
				self.layernum[1]*2,
				self.layernum[0],
				self.kernel_size,
				self.stride,
				2)
		self.outconv1 = nn.Conv1d(2*self.layernum[0], self.layernum[0],
				self.kernel_size, padding='same')
		self.out_bn = nn.BatchNorm1d(self.layernum[0])
		self.outconv2 = nn.Conv1d(self.layernum[0], self.classes, 1)

	def forward(self, x, logits=False):
		inc = self.activation(self.in_bn(self.inc(x)))
		leftskip1,d1 = self.dconv1(inc)
		leftskip2,d2 = self.dconv2(d1)
		leftskip3,d3 = self.dconv3(d2)
		leftskip4,d4 = self.dconv4(d3)
		u4 = self.uconv1(leftskip4,d4)
		u3 = self.uconv2(leftskip3,u4)
		u2 = self.uconv3(leftskip2,u3)
		u1 = self.uconv4(leftskip1,u2)
		o1 = self.activation(self.out_bn(self.outconv1(u1)))
		o2 = self.outconv2(o1)
		if logits:
			return o2
		else:
			return torch.softmax(o2, dim=1)

if __name__ == "__main__":
	from torchsummary import summary
	device = torch.device('cuda:0')
	model = torch_version_PhaseNet().to(device)
	summary(model, (3,3001))
	logits = model(torch.ones(1,3,3001).to(device), logits=False)
