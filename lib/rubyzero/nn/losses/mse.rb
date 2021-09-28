module RubyZero::NN::Losses
    class MSE < Loss
        def forward(input, target)
            err = input - target
            err = err * err
            while err.shape.size > 1
                err = err.sum(axis:0)
            end
            return err
        end
    end
end