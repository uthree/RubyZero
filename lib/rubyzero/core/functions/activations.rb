module RubyZero::Core::Functions
    class ReLU < Function
        def forward(x)
            @path_through = RubyZero::Core::Tensor.new(x.data < 0)
            @path_through.cast_to(RubyZero::FloatTensor)
            return x * @path_through
        end
        def backward(dy)
            return [ @path_through * dy ]
        end
    end
end