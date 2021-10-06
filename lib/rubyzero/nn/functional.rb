module RubyZero::NN
    module Functional
        def self.relu(x)
            return RubyZero::Core::Functions::ReLU.new().call(x)
        end

    end
end


module RubyZero
    F = RubyZero::NN::Functional
end