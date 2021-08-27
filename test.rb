require "./lib/rubyzero.rb"
t = RubyZero::Core::Tensor.new(RubyZero::Core::Shape.new(1,2,3), RubyZero::Core::DataTypes::Float32, RubyZero::Core::Device.new())
p t