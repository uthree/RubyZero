module RubyZero::Core::Functions
    class ReLU < Function
        def forward(x)
            @path_through = RubyZero::Core::Tensor.new(x.data > 0)
            @path_through.cast_to(RubyZero::FloatTensor)
            return x * @path_through
        end
        def backward(dy)
            return [ @path_through * dy ]
        end
    end

    class Sigmoid < Function
        def forward(x)
            nmath = x.device.caluculator::NMath
            data = 1.0 / (1.0 + nmath.exp(-x.data))
            return RubyZero::Core::Tensor.new(data)
        end
        def backward(dy)
            x = @inputs[0]
            return [ F.sigmoid(self.forward(x)) * (x.ones_like - F.sigmoid(self.forward(x))) * dy ]
        end
    end
end