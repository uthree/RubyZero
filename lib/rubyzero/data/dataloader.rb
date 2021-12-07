module RubyZero::Data
    class DataLoader
        def initialize(dataset, batch_size: 1, shuffle: false)
            @dataset = dataset
            @batch_size = batch_size > dataset.size ? dataset.size : batch_size
            @shuffle = shuffle
        end
        include Enumerable
        def each
            shuffled_index = nil
            shuffled_index = (0..(@dataset.length-1)).to_a.shuffle if @shuffle
            batched_index = shuffled_index.each_slice(@batch_size).to_a
            batched_index.each do |batch_idx_arr|
                datas = []
                batch_idx_arr.each do |idx|
                    datas << @dataset[idx]
                end
                # transpose 2d array
                args = datas.transpose
                args = args.map do |arg|
                    if arg.class == Array
                        if arg[0].is_a?(RubyZero::Core::Tensor)
                            next RubyZero::Core::Tensor.stack(arg)
                        elsif arg[0].is_a?(Array)
                            next RubyZero::Core::Tensor.new(arg)
                        end
                    end
                    next arg
                end
                yield(*args)
            end
        end
    end
end