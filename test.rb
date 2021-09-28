

require_relative './lib/rubyzero.rb'

include RubyZero

model = NN::Layers::MultiLayerPerceptron.new(2, 5, 1)

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

model = NN::Layers::MultiLayerPerceptron.new(2, 10, 1)
criterion = NN::Losses::MSE.new
optimizer = NN::Optimizers::SGD.new(model.parameters, lr: 0.01)
p model


100.times do |epoch|
    optimizer.zero_grad
    output = model.forward(input_data)
    loss = criterion.forward(output, output_data)
    loss.backward
    l = loss.data[0]
    puts "epoch #{epoch} | Loss : #{l}"
    optimizer.update
end

