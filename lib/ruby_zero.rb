require "numo/narray"

Dir[File.expand_path('../core/', __FILE__) << '/*.rb'].each do |file|
    require file
end

Dir[File.expand_path('../functions/', __FILE__) << '/*.rb'].each do |file|
    require file
end

Dir[File.expand_path('../optimizers/', __FILE__) << '/*.rb'].each do |file|
    require file
end

Dir[File.expand_path('../nn/', __FILE__) << '/*.rb'].each do |file|
    require file
end
