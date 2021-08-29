module RubyZero::NN
    # Model class
    class Model
        # Constructor
        def initialize(*args)
            @parameters = nil
            @children = []
        end

        # Get parameters
        # @return [Parameters]
        def parameters
            if @parameters
                return @parameters
            else 
                __init_params__()
                return @parameters
            end
        end
        # Get the output of the model
        # @return [Tensor|Array<Tensor>]
        def call(*args, **kwargs, &block)
            forward(*args, **kwargs, &block)
        end
        # Forward pass
        # @return [Tensor|Array<Tensor>]
        def forward(*args, **kwargs, &block)
            raise Core::Exceptions::NoImplementationError, "#{self.class}#forward is not implemented."
        end

        def __init_params__()
            @parameters = Parameters.new
            self.instance_variables.each do |var_key|
                next if [:@parameters, :@children].include?(var_key)
                var = instance_variable_get(var_key)
                if var.is_a? Model
                    @parameters += var.parameters
                    var.__init_params__()
                    @children << var
                elsif var.is_a? Parameters
                    @parameters += var
                elsif var.is_a? Tensor
                    @parameters << var
                end
            end
        end

        def inspect
            __init_params__() unless @parameters
            s = ''
            s += "#{self.class.name} : #{@parameters.__number_of_parameters__} parameters\n"
            @children.uniq!
            @children.each do |child|
                insp = child.inspect
                insp = insp.split("\n").map{|line| "\t" + line}.join("\n") + "\n"
                s += insp
            end
            return s
        end
    end
end