module RubyZero::Functions
    class Sigmoid < Function
        def forward(x)
            data = x.data
            @xm = x.xm
            y = 1.0 / (1.0 + @xm::NMath.exp(-data))
            return Tensor.new(y)
        end
        def backward(dy)
            x_data = @inputs[0].data
            y_dash = @xm::NMath.exp(x_data) / (@xm::NMath.exp(x_data) + 1.0)**2
            dy = dy.data * y_dash
            return [Tensor.new(dy)]
        end
    end
end