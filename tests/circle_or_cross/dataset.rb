require "../../lib/ruby_zero"
require "magro"
require "numo/narray"

def load_image_grayscale(path)
    image = Magro::IO.imread(path)
    grayscale = image.median(axis: 2)
    grayscale.cast_to(Numo::DFloat)
    grayscale /= 255.0
    grayscale = 1 - grayscale
    return grayscale
end

class ImageDataset < RubyZero::Data::Dataset
    def initialize()
        @inputs = []
        @labels = []
        # load circle images
        (1..10).each do |i|
            @inputs << load_image_grayscale("./circle_or_cross_dataset/circle/#{i}.png")
            @labels << [1, 0]
        end
        # load closs images
        (1..10).each do |i|
            @inputs << load_image_grayscale("./circle_or_cross_dataset/cross/#{i}.png")
            @labels << [0, 1]
        end
        # stack inputs and labels
        @inputs = Numo::NArray.dstack(@inputs).swapaxes(0, 2)
        @labels = Numo::NArray.dstack(@labels).swapaxes(0, 2)
    end
    def get_item(index)
        return @inputs[index, nil, nil], @labels[index, nil, nil]
    end
    def get_length()
        return @inputs.shape[0]
    end
end

