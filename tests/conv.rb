require_relative "../lib/ruby_zero"
include RubyZero
input = Tensor[
    [
        [[1], [2], [3], [4], [5]],
        [[6], [7], [8], [9], [10]],
        [[11], [12], [13], [14], [15]],
        [[16], [17], [18], [19], [20]],
        [[21], [22], [23], [24], [25]]
    ],
    [
        [[1], [2], [3], [4], [5]],
        [[6], [7], [8], [9], [10]],
        [[11], [12], [13], [14], [15]],
        [[16], [17], [18], [19], [20]],
        [[21], [22], [23], [24], [25]]
    ],
]
input.require_grad = true

conv2d = NN::Conv2d.new(1, 1, 3, stride:2)
conv2d.train
out = conv2d.call(input)
out.backward
p out.shape