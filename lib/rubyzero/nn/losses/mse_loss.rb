module RubyZero::NN::Losses
    class MSELoss < Loss
        def forward(x, target)
            F.mse_loss(x, target)
        end
    end
end
