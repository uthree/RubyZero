module RubyZero::NN
    class Model
        # Constructor
        def self.new
            super
            self.instance_variables.each do |var_key|
                var = self.instance_variable_get(var_key)
                # TODO: add to parameters
            end
        end
        def initialize
            
        end
        def call(*args, **kwargs, &block)
            forward(*args, **kwargs, &block)
        end
        def forward(*args, **kwargs, &block)
            raise Core::Exceptions::NoImplementationError, "#{self.class}#forward is not implemented."
        end
    end
end