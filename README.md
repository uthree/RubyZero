# RubyZero
A simple deep learning library for Ruby.

This library likes [PyTorch](https://github.com/pytorch/pytorch) and [DeZero](https://github.com/oreilly-japan/deep-learning-from-scratch-3).

# Example
```ruby
require_relative './lib/rubyzero.rb'

include RubyZero

class MLP < NN::Model
    def initialize(input_size, hidden_size, output_size)
        @fc1 = NN::Layers::Affine.new(input_size, hidden_size)
        @relu1 = NN::Layers::ReLU.new
        @fc2 = NN::Layers::Affine.new(hidden_size, output_size)
    end
    def forward(x)
        x = @fc1.call(x)
        x = @relu1.call(x)
        x = @fc2.call(x)
        return x
    end
end

input_data = FloatTensor[
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0]
]

output_data = FloatTensor[
    [1],
    [1],
    [0],
    [0]
]

model = MLP.new(2, 5, 1)
criterion = NN::Losses::MSE.new
optimizer = NN::Optimizers::SGD.new(model.parameters, lr: 0.01)


100.times do |epoch|
    optimizer.zero_grad
    output = model.forward(input_data)
    loss = criterion.forward(output, output_data)
    loss.backward
    l = loss.data[0]
    puts "epoch #{epoch} | Loss : #{l}"
    optimizer.update
end
```

# Installation
comming soon.

# Documentation
comming soon.
