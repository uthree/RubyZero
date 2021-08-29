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
        @direct = L::Affine.new(2, 1)
    end
    def forward(x)
        x = @direct.call(x)
        return x
    end
end

model = MLP.new(10)
optimizer = NN::Optimizers::SGD.new(model.parameters, lr: 0.1)
criterion = NN::Losses::MSELoss.new
2.times do |i|
    p "START LOOP #{i}"
    optimizer.zero_grad
    output = model.call(input_data)
    loss = criterion.call(output, target_data)
    loss.backward
    optimizer.step
    puts "#{i} loss: #{loss.data[0]}"
end
