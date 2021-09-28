module RubyZero::NN::Layers
    class MultiLayerPerceptron < ModelStack
        # @param [Array<Integer>] dims The dimensions of the input, hidden and output
        def initialize(*dims, activation: RubyZero::NN::Layers::ReLU)
            super()
            if dims.length < 2
                raise ArgumentError, "Layer must have at least 2 dimensions"
            end
            @dims = dims
            @activation_function = activation

            (@dims.size-1).times do |i|
                dim_i = @dims[i]
                dim_o = @dims[i+1]
                @models << RubyZero::NN::Layers::Affine.new(dim_i, dim_o)
                if i != @dims.size-2
                    @models << activation.new
                end
            end
        end
    end
end