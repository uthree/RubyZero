require_relative './lib/rubyzero.rb'

include RubyZero

mlp = L.mlp(2, 10, 1)
data = RubyZero::Data::Presets::Xor.new
trainer = Utils::Trainer.new(mlp)
p trainer.train(data, data, num_epochs: 1000)
