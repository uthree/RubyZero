require_relative "./losses.rb"

module RubyZero::Losses
    class MeanSquaredError < LossFunction
        def initialize()
            super()
        end
        def forward(y, y_pred)
            err = (y - y_pred)
            err = RubyZero::Functions::Square.new().call(err)
            #p err.data
            while err.ndim > 0
                err = err.mean()
            end
            #p err.data
            return err
        end
    end
end