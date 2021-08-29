module RubyZero::NN::Layers
    # Affine Layer
    # This layer is calculates the affine transformation of the input
    # y = wx + b
    class Affine < Layer
        def initialize(input_size, output_size, bias: true)
            super
            @weight = FloatTensor.rand_norm(output_size, input_size)
            if bias
                @bias = FloatTensor.rand_norm(output_size)
            end
        end

        def forward(input)
            input = input.swap_axes(0, 1)
            output = @weight.dot(input).swap_axes(0,1)
            output += @bias if @bias
            return output
        end
    end
end