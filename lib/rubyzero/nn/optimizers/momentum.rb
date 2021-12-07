module RubyZero::NN::Optimizers
    class Momentum < Optimizer
        def initialize(parameters, lr: 0.001, alpha: 0.9)
            @lr = lr
            @parameters = parameters
            @alpha = alpha
            @velocities = []
        end
        def update()
            @parameters.each_with_index do |tensor, index|
                if @velocities[index].nil?
                    @velocities[index] = 0
                else
                    vel = @velocities[index] * @alpha
                    uparam = tensor.grad.data + vel
                    tensor.data -= uparam * @lr
                end
            end
        end
        alias :step :update
    end
end