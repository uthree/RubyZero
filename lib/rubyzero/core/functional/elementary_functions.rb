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
end