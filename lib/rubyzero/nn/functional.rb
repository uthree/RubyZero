module RubyZero::NN
    module Functional
        def self.relu(x)
            return RubyZero::Core::Functions::ReLU.new().call(x)
        end
        def self.log(x)
            return RubyZero::Core::Functions::Log.new().call(x)
        end
        def self.sigmoid(x)
            return RubyZero::Core::Functions::Sigmoid.new().call(x)
        end
        def self.exp(x)
            return RubyZero::Core::Functions::Exp.new().call(x)
        end
        def self.sqrt(x)
            return RubyZero::Core::Functions::SquareRoot.new().call(x)
        end
    end
end


module RubyZero
    F = RubyZero::NN::Functional
end