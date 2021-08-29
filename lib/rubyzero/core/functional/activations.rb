module RubyZero::Core::Functional
    # @param [Tensor] x
    # @return [Tensor]
    def self.sigmoid(x, alpha: 1.0)
        Core::Functions::Sigmoid.new(alpha: alpha).call(x)
    end

    # @param [Tensor] x
    # @return [Tensor]
    def self.relu(x)
        Core::Functions::ReLU.new.call(x)
    end

    def self.softmax(x, axis: nil)
        axis = x.shape.axes[-1] if axis.nil?
        x = x.swap_axes(axis, 0) if axis != 0
        x = Core::Functions::SoftmaxZeroAxis.new.call(x)
        x = x.swap_axes(0, axis) if axis != 0
        return x
    end
end