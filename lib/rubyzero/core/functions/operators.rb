module RubyZero::Core::Functions
    class Add < Function
        def forward(x1, x2)
            new_arr = x1.data + x2.data
            new_t = RubyZero::Core::Tensor.new(new_arr, device: x1.device)
            return new_t
        end
        def backward(dy)
            return dy, dy
        end
    end

    class Sub < Function
        def forward(x1, x2)
            new_arr = x1.data - x2.data
            new_t = RubyZero::Core::Tensor.new(new_arr, device: x1.device)
            return new_t
        end
        def backward(dy)
            return dy, -dy
        end
    end

    class Mul < Function
        def forward(x1, x2)
            new_arr = x1.data * x2.data
            new_t = RubyZero::Core::Tensor.new(new_arr, device: x1.device)
            return new_t
        end
        def backward(dy)
            x1, x2 = @inputs
            return dy * x2, dy * x1
        end
    end

    class Div < Function
        def forward(x1, x2)
            new_arr = x1.data / x2.data
            new_t = RubyZero::Core::Tensor.new(new_arr, device: x1.device)
            return new_t
        end
        def backward(dy)
            x1, x2 = @inputs
            return dy / x2, -dy * x1 / x2 ** 2
        end
    end

    class Pow < Function
        def forward(x1, x2)
            new_arr = x1.data ** x2.data
            new_t = RubyZero::Core::Tensor.new(new_arr, device: x1.device)
            return new_t
        end
        def backward(dy) 
            x1, x2 = @inputs
            return dy * x2 * x1 ** (x2 - 1), dy * x1 ** x2 * Log.new().call(x1)
        end
    end

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

    class MulScalar < Function
        def initialize(scalar)
            @scalar = scalar
        end
        def forward(x1)
            new_arr = x1.data * @scalar
            new_t = RubyZero::Core::Tensor.new(new_arr, device: x1.device)
            return new_t
        end
        def backward(dy)
            return [dy * @scalar]
        end
    end

    class DivScalar < Function
        def initialize(scalar)
            @scalar = scalar
        end
        def forward(x1)
            new_arr = x1.data / @scalar
            new_t = RubyZero::Core::Tensor.new(new_arr, device: x1.device)
            return new_t
        end
        def backward(dy)
            return [dy / @scalar]
        end
    end
end