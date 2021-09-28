module RubyZero::NN::Layers
    # ReLU Layer
    class ReLU < Layer
        def initialize()
        end
        def forward(x)
            return RubyZero::Core::Functions::ReLU.new().call(x)
        end
    end
end