require_relative "../lib/ruby_zero"
include RubyZero

def im2col_1d(x, kernel, padding, stride)

end

def padding_1d(x, padding) # x: [batch, width, channels]
    length = x.shape[1]
    padded_len = length + 2 * padding
    new_shape = [x.shape[0], padded_len, x.shape[2]]
    t = Tensor.zeros(new_shape, x.dtype)
    t[nil, padding-1..length-1, nil] = x
    return t
end

tensor = FloatTensor.rand([3, 5, 2])
p tensor
tensor = padding_1d(tensor, 1)
p tensor
