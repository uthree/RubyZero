module RubyZero::Core::Functional
    # @param [Tensor] x1
    # @param [Tensor] x2
    # @return [Tensor]
    def self.matmul(x1, x2)
        return Core::Functions::MatMul.new.call(x1, x2)
    end
end