require_relative "../lib/ruby_zero"
include RubyZero

def im2col_1d(x, kernel, padding, stride)

end

def padding_1d(x, padding) # x: [batch, width, channels]
    length = x.shape[1]
    padded_len = length + 2 * padding
    new_shape = [x.shape[0], padded_len, x.shape[2]]
    Tensor.zeros
end

p FloatTensor.zeros(1)