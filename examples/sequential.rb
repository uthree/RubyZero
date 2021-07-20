require "../lib/ruby_zero.rb"
include RubyZero

model = NN::Sequential.new
model << NN::Linear.new(2, 10)
model << NN::ReLU.new
model << NN::Linear.new(10, 10)
model << NN::Sigmoid.new
model << NN::Linear.new(10, 1)

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

optimizer = Optimizers::SGD.new(learning_rate:0.01)
criterion = Losses::MeanSquaredError.new
optimizer << model
model.train
10000.times do 
    optimizer.zero_grad
    out = model.call(input)
    loss = criterion.call(out, target)
    loss.backward
    optimizer.step
    puts "Loss: #{loss.item.round(5)}"
end

model.eval()
p model.call(input)

