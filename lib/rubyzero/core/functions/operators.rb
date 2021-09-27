module RubyZero::Core::Functions
    class Add < Function
        def initialize()
        end
        def forward(x1, x2)
            new_arr = x1.data + x2.data
            new_t = RubyZero::Core::Tensor.new(new_arr, device: x1.device)
            return new_t
        end
        def backward(dy)
            return dy, dy
        end
    end
end