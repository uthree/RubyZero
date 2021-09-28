module RubyZero::Data::Presets
    class Xor < RubyZero::Data::Dataset
        def initialize
            @inputs = [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1]
            ]
            @targets = [
                [0],
                [1],
                [1],
                [0]
            ]
        end
        def getitem(idx)
            return @inputs[idx], @targets[idx]
        end
        def length
            return 4
        end
    end
end