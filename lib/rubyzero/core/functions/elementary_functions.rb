module RubyZero::Core::Functions
    class Log < Function
        def forward(x1)
            nmath = x1.device.xmo::NMath
            new_arr = nmath.log(x1.data)
            new_t = RubyZero::Core::Tensor.new(new_arr, device: x1.device)
            return new_t
        end

        def backward(dy)
            x1 = @inputs[0]
            return [dy / x1]
        end
    end

    class Exp < Function
        def forward(x1)
            nmath = x1.device.xmo::NMath
            new_arr = nmath.exp(x1.data)
            new_t = RubyZero::Core::Tensor.new(new_arr, device: x1.device)
            return new_t
        end

        def backward(dy)
            x1 = @inputs[0]
            return [dy * Exp.new.call(x1)]
        end
    end

    class SquareRoot < Function
        def forward(x1)
            nmath = x1.device.xmo::NMath
            new_arr = nmath.sqrt(x1.data)
            new_t = RubyZero::Core::Tensor.new(new_arr, device: x1.device)
            return new_t
        end
        def backward(dy)
            x1 = @inputs[0]
            return [dy / (SquareRoot.new.call(x1 * 2))]
        end
    end
end