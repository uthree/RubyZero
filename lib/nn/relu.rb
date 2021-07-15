module RubyZero::NN
    class ReLU < Module
        def initialize()
            @f = RubyZero::Functions::ReLU.new
        end
        def forward(input)
            @f.call(input)
        end
    end
end