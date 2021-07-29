module RubyZero
    module Data
        
    end
end

module RubyZero::Data
    class Dataset
        public
        def initialize()

        end

        def [](index)
            return get_item(index)
        end
        def length
            return get_length
        end
        def size
            return get_length
        end
        private
        def get_item(idx)
            raise NotImplementedError, "Not Implemented #{self.calss}.getitem_index"
        end
        def get_length
            raise NotImplementedError, "Not Implemented #{self.class}.get_length"
        end
    end

    class DataLoader
        public
        include Enumerable
        def initialize(dataset, batch_size=1, shuffle=false)
            @dataset = dataset
            @batch_size = batch_size
            @shuffle = shuffle
            @length = @dataset.size
            @indexes = 0.upto(@length - 1).to_a
        end

        def each(*args, &block)
            if block
                if @shuffle
                    @indexes = @indexes.shuffle
                end
                b = (@length.to_f / @batch_size).ceil
                b.times do |now_batch_num|
                    batch = get_item_batch(now_batch_num)
                    new_batch = Array.new(batch[0].length, [])
                    new_batch.each_with_index do |n, idx|
                        new_batch[idx] = batch.map{|b| b[idx]}
                    end
                    block.call(new_batch)
                end
            else
                return Enumerator.new(self, :each, *args)
            end
        end

        private
        def get_item_batch(now_batch_num)
            access_indexes = @indexes.slice(now_batch_num * @batch_size, @batch_size)
            data = []
            access_indexes.each do |index|
                data << @dataset.get_item(index)
            end
            return data
        end
    end
end


