module RubyZero::NN::Layers
    class MultiLayerPerceptron < ModelStack
        # @option opts [Integer] :dims The dimensions of the input, hidden and output
        # 
        # @note [5, 3] -> 5 inputs, 3 outputs
        #   [5, 3, 2] -> 5 inputs, 3 hidden, 2 outputs
        # 
        def initialize(*dims, activation: RubyZero::NN::Layers::ReLU, normalization: RubyZero::NN::Layers::BatchNormalization)
            super()
            if dims.length < 2
                raise ArgumentError, "Must have two layers or more."
            end
            @dims = dims
            @activation_function = activation

            (@dims.size-1).times do |i|
                dim_i = @dims[i]
                dim_o = @dims[i+1]
                @models << RubyZero::NN::Layers::Affine.new(dim_i, dim_o)
                if i != @dims.size-2
                    @models << activation.new
                    if normalization
                        @models << normalization.new
                    end
                end
            end
        end
    end
end