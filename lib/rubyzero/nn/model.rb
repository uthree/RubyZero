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
        def parameters
            param_keys = instance_variables
            params = []
            param_keys.each do |key|
                obj = instance_variable_get(key)
                if obj.is_a?(RubyZero::Core::Tensor)
                    obj.requires_grad = true
                    params << obj
                elsif obj.is_a?(RubyZero::NN::Parameters)
                    params += obj.elements
                else obj.is_a?(RubyZero::NN::Model)
                    params += obj.parameters.elements
                end
            end
            return Parameters.new(params)
        end
        def __get_str__(num_indents)
            keys = instance_variables
            children = []
            keys.each do |key|
                obj = instance_variable_get(key)
                if obj.is_a?(RubyZero::NN::Model)
                    children << obj
                end
            end
            indents = "  " * num_indents
            s = "#{indents}#{self.class.name} #{parameters.size} params\n"
            children.each do |child|
                s += child.__get_str__(num_indents + 1)
            end
            return s
        end
        def inspect
            return __get_str__(0)
        end
        def train()
            self.parameters.map do |param|
                param.requires_grad = true
            end
            return self
        end
        def eval()
            def train()
                self.parameters.map do |param|
                    param.requires_grad = false
                end
                return self
            end
        end
    end
end