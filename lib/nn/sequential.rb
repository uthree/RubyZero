require_relative "./nn.rb"

module RubyZero::NN
    class ModuleList < Module
        def initialize
            @modules = []
            super
        end
        def __update_childlen__()
            @__parameters__ = Parameters.new
            inv = instance_variables
            inv.delete(:@__parameters__)
            inv.delete(:@__childlen__)
            inv.delete(:@__flag_init_update__)

            inv.each do |k|
                value = instance_variable_get(k)
                if value.is_a?(RubyZero::Tensor)
                    @__parameters__ << value if value.trainable?
                end
            end

            inv.each do |k|
                child = instance_variable_get(k)
                @__childlen__ << child if child.kind_of?(Module)
            end

            @modules.each do |m|
                @__childlen__ << m if m.kind_of?(Module)
            end
            
            @__childlen__.each do |child|
                #p child
                child.__update_childlen__
                @__parameters__ += child.parameters
            end

            @__flag_init_update__ = true
        end
        def <<(other_module)
            raise InvalidArgumentError, "ModuleList can only accept Module" unless other_module.kind_of?(Module)
            @modules << other_module
            return self
        end
        def modules
            return @modules
        end
    end

    class Sequential < Module
        def initialize
            @mods = ModuleList.new
            super
        end
        def forward(*inputs)
            @mods.modules.each do |m|
                output = m.call(*inputs)
                if output.is_a?(Array)
                    inputs = output
                else
                    inputs = [output]
                end
            end
            if inputs.size > 1
                return *inputs
            else
                return inputs.first
            end
        end
        def <<(other_module)
            @mods << other_module
        end
    end
end

