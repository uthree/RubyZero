require_relative "./nn.rb"

module RubyZero::NN
    class Dropout < Module
        def initialize(p=0.5)
            @p = p
            super
        end
        def forward(x)
            #TODO: Implement dropout
        end
    end
end