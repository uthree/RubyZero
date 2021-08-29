module RubyZero::Core::Functional
    # Returns the Mean Squared Error (MSE) between the two given arrays.
    # @param [Tensor] input
    # @param [Tensor] target
    # @return [Tensor]
    def self.mse_loss(input, target)
        # flatten the input and target tensors
        input, target = input.reshape(input.shape.to_a.inject(:*)), target.reshape(target.shape.to_a.inject(:*))
        # calculate the mean squared error
        ((input - target) ** 2).mean
    end
end