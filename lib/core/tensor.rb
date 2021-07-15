module RubyZero
    class Tensor
        attr_reader :xm, :dtype
        attr_accessor :data, :grad, :require_grad, :grad_fn, :generation
        def initialize(data, require_grad:true, grad:nil, generation:0, xm:Numo, dtype:nil)
            if data.is_a? xm::NArray # NArray系列の場合はそのままデータに代入
                @data = data
            elsif data.is_a?(Array) # 配列の場合はNumo::NArrayに変換
                if dtype.nil?
                    @data = xm::NArray[*data]
                else
                    @data = dtype.get_type(xm)[*data]
                end
            elsif data.is_a?(Tensor) # Tensorの場合はコピーしてデータに代入
                @data = data.data.dup
            elsif data.is_a?(Integer) # 整数の場合はNumo::NArrayに変換
                @data = xm::NArray[data]
            elsif data.is_a?(Float) # 浮動小数点数の場合はNumo::NArrayに変換
                @data = xm::NArray[data]
            elsif data.is_a?(Complex) # 複素数の場合はNumo::NArrayに変換
                @data = xm::NArray[data]
            else # 変換できないのでエラーにする
                raise InvalidValueError, "unsupported type #{data.class} for #{self.class}"
            end
            
            @xm = xm
            update_dtype()
            @grad_fn = nil
            @grad = nil
            @generation = generation
            @require_grad = require_grad
        end

        # 勾配を追加する
        def add_grad(other_tensor, require_grad:false)
            @grad ||= zeros_like
            @grad.require_grad = require_grad
            @grad += other_tensor
        end

        # dtypeを更新する。
        def update_dtype()
            @dtype = convert_to_rubyzero_dtype(@xm, @data.class)
        end

        # initialize with bracket
        def self.[](*data)
            Tensor.new(data)
        end

        # initialize zeros with same shape
        def zeros_like
            data = self.dtype.get_type(self.xm).zeros(*self.shape)
            Tensor.new(data, require_grad:@require_grad, grad:@grad, generation:@generation, xm:@xm, dtype:@dtype)
        end

        # initialize zeros with same shape
        def ones_like
            data = self.dtype.get_type(self.xm).ones(*self.shape)
            Tensor.new(data, require_grad:@require_grad, grad:@grad, generation:@generation, xm:@xm, dtype:@dtype)
        end
        
        def shape
            return @data.shape
        end

        def ndim
            return @data.ndim
        end

        def reshape(*shape)
            return Functions::ReShape.new(*shape).call(self)
        end

        def transpose(*axes)
            return Functions::Transpose.new(*axes).call(self)
        end

        def swap_axes(*axes)
            return Functions::SwapAxes.new(*axes).call(self)
        end

        def repeat(repeats, axis:0)
            # どの軸にもRepeatできるように拡張したもの
            shape=self.shape.dup.insert(0, repeats)
            target_shape = self.shape.dup.insert(axis, repeats)
            return Functions::RepeatZeroAxis.new(repeats).call(self.swap_axes(axis, 0)).reshape(*target_shape).swap_axes(0, axis)
        end

        def inspect
            return "#< RubyZero::Tensor dtype=#{dtype} shape=#{shape} grad_fn=#{@grad_fn.class} \n (#{@data.inspect})>"
        end
    end
end