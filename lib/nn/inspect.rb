module RubyZero::NN
    class Module
        def inspect()
            __inspect__()
        end
        
        def __inspect__(nest=0)
            self.__update_childlen__
            s = ""
            if nest == 0
                s += "-"*60 + "\n"
                s += "summary of #{self.class}\n"
                s += "-"*60 + "\n"
            end
            
            tabs = " " * nest
            num_param = 0
            parameters.elements.each do |param|
                num_param += param.shape.inject(:*)
            end
            n = nest+0.5
            s += "#{tabs}#{self.class.name} : #{num_param}\n" unless nest == 0
            @__childlen__.each do |child|
                s += "#{tabs}#{child.__inspect__(n)}"
            end

            if nest == 0
                s += "-"*60 + "\n"
                s += "parameters: #{num_param}\n"
                s += "-"*60 + "\n"
            end
            return s
        end
    end
end