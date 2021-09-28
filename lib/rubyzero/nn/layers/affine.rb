module RubyZero::NN::Layers
    # Affine layer
    class Affine < Layer
        # @param input_size [Integer] input size
        # @param output_size [Integer] output size
        # @param bias [Boolean] whether to use bias
        def initialize(input_size, output_size, bias: true)
            @weight = RubyZero::Float32.randn([input_size, output_size])
            @bias = RubyZero::Float32.randn([output_size]) if bias
        end
        # Forward pass
        # @param x [RubyZero::Float32] input
        # @return [RubyZero::Float32] output
        def forward(x)
            x = x.dot(@weight)
            if @bias
                x += @bias.repeat(x.shape[0], axis: 0)
            end
        end
    end
end