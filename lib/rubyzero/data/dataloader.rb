module RubyZero::Data
    class DataLoader
        def initialize(dataset, batch_size: 1, shuffle: false)
            @dataset = dataset
            @batch_size = batch_size > dataset.size ? dataset_size : batch_size
            @shuffle = shuffle
        end
        include Enumerable
        def each(&block)
            shuffled_index = nil
            shuffled_index = (0..(@dataset.length-1)).to_a.shuffle if @shuffle
            batched_index = shuffled_index.each_slice(@batch_size).to_a
            batched_index.each do |batch_idx_arr|
                datas = []
                batch_idx_arr.each do |idx|
                    datas << @dataset[idx]
                end
                if datas[0].class.is_a?(RubyZero::Core::Tensor)
                    # TODO: convert to tensor
                end
            end
        end

    end
end