module RubyZero::NN
    class ReLU < Module
        def initialize()
            @f = RubyZero::Functions::ReLU.new
            super
        end
        def forward(input)
            @f.call(input)
        end
    end
end