module RubyZero::NN::Layers
    class Affine < Layer
        def initialize(input_size, output_size, bias: true)
            @weight = RubyZero::Float64.randn([input_size, output_size])
            @bias = RubyZero::Float64.randn([output_size]) if bias
        end
        def forward(x)
            x = x.dot(@weight)
            if @bias
                x += @bias.repeat(x.shape[0], axis: 0)
            end
        end
    end
end