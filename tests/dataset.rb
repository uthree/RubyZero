require "../lib/ruby_zero"


class MyDataSet < RubyZero::Data::Dataset
    def initialize(ln)
        @length = ln
    end
    def get_item(idx)
        return idx+1, rand(100)
    end
    def get_length
        return @length
    end
end
ds = MyDataSet.new(10)

dl = RubyZero::Data::DataLoader.new(ds)
dl.each do |input, label|
    p input
    p label
end