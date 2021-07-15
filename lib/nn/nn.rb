module RubyZero
    module NN

    end
end

module RubyZero::NN
    class Module # template module of neural network
        def initialize(*args, **kwargs, &block)
            @parameters = Parameters.new
            @childlen = []
        end
        def call(*args, **kwargs, &block)
            forward(args, kwargs, &block)
        end
        def forward(*args, **kwargs, &block)
            raise NotImplementedError, "#{self.class}.forward method not implemented"
        end
        def eval()
            @parameters.eval
        end
        def train()
            @parameters.train
        end
    end
end