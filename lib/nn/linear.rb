require_relative "./nn.rb"

module RubyZero::NN
    class Linear < Module
        def initialize(input_size, output_size, bias:true)
            @weight = FloatTensor.rand_norm([input_size, output_size])
            @weight.trainable = true
            if bias
                @bias = FloatTensor.rand_norm([output_size])
                @bias.trainable = true
            end
            super()
        end
        def forward(x)
            y = x.dot(@weight)
            if @bias
                b = @bias.repeat(x.shape[0], axis:0)
                #p x.shape[0], b.shape
                y += b
            end
            return y
        end
    end
end