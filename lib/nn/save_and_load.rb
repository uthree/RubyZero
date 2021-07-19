require_relative "./nn.rb"
require "yaml"

module RubyZero::NN
    class Module
        def save(path)
            File.open(path, "w+") do |f|
                YAML.dump(self, f)
            end
        end
    end
end

module RubyZero
    def load(path)
        return YAML.load_file(path)
    end
end