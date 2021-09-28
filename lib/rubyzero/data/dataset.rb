module RubyZero::Data
    class Dataset
        def initialize()

        end
        def getitem(index)

        end
        def length()

        end
        def [](index)
            getitem(index)
        end
        alias :size :length
    end
end