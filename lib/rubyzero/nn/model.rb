module RubyZero::NN
    # Model class
    class Model
        def initialize

        end
        def forward
            raise RubyZero::Core::Exceptions::NoInplementError, "#{self.class}.forward() method is not implemented"
        end
        def call(*args)
            return forward(*args)
        end
    end
end