module RubyZero::Core::Functional
    # @param [Tensor] x
    # @retirn [Tensor]
    def sigmoid(x, alpha: 1.0)
        Functions::Sigmoid.new(alpha: alpha).call(x)
    end

    def relu(x)
        Functions::Relu.new.call(x)
    end
end