require "./lib/ruby_zero.rb"
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
        @f1 = NN::ReLU.new
        @l2 = NN::Linear.new(mid_units,1)
        @f2 = NN::ReLU.new
        super()
    end
    def forward(x)
        x = @f1.call(@l1.call(x))
        x = @f2.call(@l2.call(x))
        return x
    end
end

model = TwoLP.new()
optimizer = Optimizers::SGD.new()
optimizer << model
criterion = NN::MeanSquaredError.new()
model.train

num_epochs = 1
num_epochs.times do |epoch|
    optimizer.init_gradients
    out = model.call(input)
    loss = criterion.call(out, target)
    loss.backward()
    optimizer.step()
    #p loss
end