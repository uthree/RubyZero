require "./lib/rubyzero.rb"
include RubyZero

input_data = FloatTensor[
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

target_data = FloatTensor[
    [0],
    [1],
    [1],
    [0]
]

class MLP < NN::Model
    def initialize(mid_units)
        super
        @affine1 = L::Affine.new(2, mid_units)
        @relu1 = L::ReLU.new
        @affine2 = L::Affine.new(mid_units, 1)
    end
    def forward(x)
        x = @affine1.call(x)
        x = @relu1.call(x)
        x = @affine2.call(x)
        return x
    end
end

model = MLP.new(10)
optimizer = NN::Optimizers::SGD.new(model.parameters, lr: 0.1)
criterion = NN::Losses::MSELoss.new
10.times do |i|
    optimizer.zero_grad
    output = model.call(input_data)
    loss = criterion.call(output, target_data)
    loss.backward
    optimizer.step
    puts "loss: #{loss.data}"
end
