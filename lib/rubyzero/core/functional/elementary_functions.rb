module RubyZero::Core::Functional
    # logarithm function (y = log(e, x))
    # @param [Tensor] x
    # @return [Tensor]
    def log(x)
        return Functions::Log.new().call(x)
    end
    # Sine function (y = sin(x))
    # @param [Tensor] x
    # @return [Tensor]
    def sin(x)
        return Functions::Sin.new().call(x)
    end
    # Cosine function (y = cos(x))
    # @param [Tensor] x
    # @return [Tensor]
    def cos(x)
        return Functions::Cos.new().call(x)
    end
    # Tangent function (y = tan(x))
    # @param [Tensor] x
    # @return [Tensor]
    def tan(x)
        return Functions::Tan.new().call(x)
    end
    # Exponential function (y = e^x)
    # @param [Tensor] x
    # @return [Tensor]
    def exp(x)
        return Functions::Exp.new().call(x)
    end
    # Raise to power function (y = x1, x2)
    # @param [Tensor] x1
    # @param [Tensor] x2
    # @return [Tensor]
    def pow(x1, x2)
        return Functions::Pow.new().call(x1, x2)
    end
    # Sine hyperbolic function (y = sinh(x))
    # @param [Tensor] x
    # @return [Tensor]
    def sinh(x)
        return Functions::Sinh.new().call(x)
    end
    # Cosine hyperbolic function (y = cosh(x))
    # @param [Tensor] x
    # @return [Tensor]
    def cosh(x)
        return Functions::Cosh.new().call(x)
    end
    # Tangent hyperbolic function (y = tanh(x))
    # @param [Tensor] x
    # @return [Tensor]
    def tanh(x)
        return Functions::Tanh.new().call(x)
    end
end