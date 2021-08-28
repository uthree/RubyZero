module RubyZero::Core::Functional
    # logarithm function (y = log(e, x))
    # @param [Tensor] x
    # @return [Tensor]
    def self.log(x)
        return Core::Functions::Log.new().call(x)
    end
    # Sine function (y = sin(x))
    # @param [Tensor] x
    # @return [Tensor]
    def self.sin(x)
        return Core::Functions::Sin.new().call(x)
    end
    # Cosine function (y = cos(x))
    # @param [Tensor] x
    # @return [Tensor]
    def self.cos(x)
        return Core::Functions::Cos.new().call(x)
    end
    # Tangent function (y = tan(x))
    # @param [Tensor] x
    # @return [Tensor]
    def self.tan(x)
        return Core::Functions::Tan.new().call(x)
    end
    # Exponential function (y = e^x)
    # @param [Tensor] x
    # @return [Tensor]
    def self.exp(x)
        return Core::Functions::Exp.new().call(x)
    end
    # Raise to power function (y = x1, x2)
    # @param [Tensor] x1
    # @param [Tensor] x2
    # @return [Tensor]
    def self.pow(x1, x2)
        return Core::Functions::Pow.new().call(x1, x2)
    end
    # Sine hyperbolic function (y = sinh(x))
    # @param [Tensor] x
    # @return [Tensor]
    def self.sinh(x)
        return Core::Functions::Sinh.new().call(x)
    end
    # Cosine hyperbolic function (y = cosh(x))
    # @param [Tensor] x
    # @return [Tensor]
    def self.cosh(x)
        return Core::Functions::Cosh.new().call(x)
    end
    # Tangent hyperbolic function (y = tanh(x))
    # @param [Tensor] x
    # @return [Tensor]
    def self.tanh(x)
        return Core::Functions::Tanh.new().call(x)
    end
end