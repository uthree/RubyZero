
require_relative './lib/rubyzero.rb'

dataset = RubyZero::Data::Presets::Xor.new
dataloader = RubyZero::Data::DataLoader.new(dataset, batch_size: 4, shuffle: true)
model = RubyZero::NN::Layers::MultiLayerPerceptron.new(2,100,1)
criterion = RubyZero::NN::Losses::MSE.new
optimizer = RubyZero::NN::Optimizers::SGD.new(model.parameters, lr: 0.00001)

1000.times do |i|
    dataloader.each do |inputs, targets|
        optimizer.zero_grad
        outputs = model.forward(inputs)
        loss = criterion.forward(outputs, targets)
        l = loss.data[0]
        puts "Epoch #{i} | Loss: #{l}"
        loss.backward
        optimizer.step
    end
end