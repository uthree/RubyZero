module RubyZero::NN
    class MeanSquaredError < Module
        def forward(output, target)
            error = (output - target)
            error = error * error
            return error.sum.sum
        end
    end
end