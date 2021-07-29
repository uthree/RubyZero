require "../../lib/ruby_zero"
require "./dataset.rb"
include RubyZero

dataset = ImageDataset.new
dataloader = RubyZero::Data::DataLoader.new(dataset, batch_size=20, shuffle=true)

model = NN::Sequential.new()
model << NN::Linear.new(784, 100)
model << NN::Sigmoid.new()
model << NN::Linear.new(100, 2)

criterion = Losses::MeanSquaredError.new()
optimizer = Optimizers::SGD.new(learning_rate: 0.0001)
optimizer << model


1000.times do |epoch| # epoch loop
    epoch_losses = 0
    dataloader.each do |input, label| # batch loop
        input_tensor = Tensor.new(input)
        input_tensor = input_tensor.reshape(input_tensor.shape[0], 784)
        label_tensor = Tensor.new(label)
        label_tensor = label_tensor.reshape(label_tensor.shape[0], 2)
        
        optimizer.zero_grad
        output = model.call(input_tensor)
        loss = criterion.call(output, label_tensor)
        loss.backward
        optimizer.step
        epoch_losses += loss.item
    end
    puts "#{epoch} loss: #{epoch_losses}"
end