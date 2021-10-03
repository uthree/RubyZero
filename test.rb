require_relative './lib/rubyzero.rb'

dataset = RubyZero::Data::Presets::Xor.new
model = RubyZero::NN::Layers::MultiLayerPerceptron.new(2,10,10,1)
criterion = RubyZero::NN::Losses::MSE.new
optimizer = RubyZero::NN::Optimizers::Momentum.new(model.parameters)

t = RubyZero::Utils::Trainer.new(model, loss_function: criterion, optimizer: optimizer)
t.train(dataset, dataset, num_epochs: 20, batch_size:4)