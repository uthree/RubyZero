require_relative "./losses.rb"

module RubyZero::Losses
    class MeanSquaredError < LossFunction
        def initialize()
            super()
        end
        def forward(y, y_pred)
            err = (y - y_pred)
            err = err * err
            while err.ndim > 1
                err = err.mean()
            end
            return err
        end
    end
end