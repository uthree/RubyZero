module RubyZero::Core::Functions
    class Log < Function
        # Take the logarithm of a tensor
        # @param [Tensor] x
        def forward(x)
            x.cast_to(DoubleTensor)
            calculator = x.device.calculator
            data = calculator::NMath.log(x.data)
            return Tensor.new(data)
        end
        def backward(dy)
            return [dy / @input[0]]
        end
    end
    class Sin < Function
        # Take the sine of a tensor
        # @param [Tensor] x
        # @return [Tensor]
        def forward(x)
            x.cast_to(DoubleTensor)
            calculator = x.device.calculator
            data = calculator::NMath.sin(x.data)
            return Tensor.new(data)
        end
        def backward(dy)
            return [dy * F.cos(@input[0])]
        end
    end
    class Cos < Function
        # Take the cosine of a tensor
        # @param [Tensor] x
        # @return [Tensor]
        def forward(x)
            x.cast_to(DoubleTensor)
            calculator = x.device.calculator
            data = calculator::NMath.cos(x.data)
            return Tensor.new(data)
        end
        def backward(dy)
            return [dy * F.sin(@input[0])]
        end
    end
    class Tan < Function
        # Take the tangent of a tensor
        # @param [Tensor] x
        # @return [Tensor]
        def forward(x)
            x.cast_to(DoubleTensor)
            calculator = x.device.calculator
            data = calculator::NMath.tan(x.data)
            return Tensor.new(data)
        end
        def backward(dy)
            return [dy / F.pow(F.cos(@input[0]), 2)]
        end
    end
    class Exp < Function
        # Take the exponential of a tensor
        # @param [Tensor] x
        # @return [Tensor]
        def forward(x)
            x.cast_to(DoubleTensor)
            calculator = x.device.calculator
            data = calculator::NMath.exp(x.data)
            return Tensor.new(data)
        end
        def backward(dy)
            return [dy * F.exp(@input[0])]
        end
    end
end