require "../lib/ruby_zero.rb"
include RubyZero

input = FloatTensor[
    [0,0],
    [1,0],
    [0,1],
    [1,1]
]

target = FloatTensor[
    [0],
    [1],
    [1],
    [0]
]


class TwoLP < NN::Module
    def initialize(mid_units = 5)
        @l1 = NN::Linear.new(2,mid_units)
        @f1 = NN::Sigmoid.new
        @l2 = NN::Linear.new(mid_units,1)
        super()
    end
    def forward(x)
        x = @f1.call @l1.call(x)
        x = @l2.call(x)
        return x
    end
end

model = TwoLP.new(mid_units = 10)
criterion = Losses::MeanSquaredError.new
optimizer = Optimizers::SGD.new(learning_rate=0.01)
optimizer << model
model.train

output = nil

1000.times do
    optimizer.zero_grad()
    output = model.call(input)
    loss = criterion.call(output,target)
    loss.backward
    optimizer.step()
end

p output.data