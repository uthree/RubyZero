require_relative './lib/rubyzero.rb'

dataset = RubyZero::Data::Presets::Xor.new
model = RubyZero::NN::Layers::MultiLayerPerceptron.new(2,10,10,1)
criterion = RubyZero::NN::Losses::MSE.new
optimizer = RubyZero::NN::Optimizers::SGD.new(model.parameters, lr: 0.001)

t = RubyZero::Utils::Trainer.new(model, criterion, optimizer)
t.train(dataset, dataset, num_epochs: 1000, batch_size:10)