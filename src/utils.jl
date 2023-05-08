
function arraytoRGB(imgm::Array{Float32,3})
    [ RGB{N0f8}(imgm[1,i,j],imgm[2,i,j],imgm[3,i,j]) for i in 1:size(imgm,2), j in 1:size(imgm,3) ]
end

function arraytoRGB(imgm::Array{UInt8,3})
    arraytoRGB( imgm./255f0 )
end




function standardizevideo!(frames::Array{Float32}; shift::Float32=0.5f0, σ::Float32=3f0, absolute::Bool=false, shows::Bool=false)
    m = mean(frames)
    s = std(frames)
    frames .-= m             # memory efficient in place broadcast!
    if shows println("s=$s") end
    if absolute mult = 1f0 ./σ else mult = σ*s end
    if s>1f-6 frames ./= mult end
    frames .+= shift
    clamp!(frames,0f0,1f0)
end

function standardizeframe!(frame::Matrix{UInt8}; σ=3.)
    aux = Float32.(frame)
    m = mean(aux)
    s = std(aux)
    aux .-= m
    aux ./= σ*s
    aux .+= 0.5f0
    aux .*= 255
    frame[:,:] = converttouint8.(aux)
end

function standardizeframe!(frame::Matrix{Gray{N0f8}}; σ=3.)
    aux = Float32.(frame)
    m = mean(aux)
    s = std(aux)
    aux .-= m
    aux ./= σ*s
    aux .+= 0.5f0
    frame[:,:] = Gray.(converttoN0f8.(aux))
end



function squeezevideo!(frames::Array{Float32})
    min = minimum(frames)
    max = maximum(frames)
    frames .-=  min             # memory efficient in place broadcast!
    frames ./= (max - min)
end

function squeezeframe!(frame::Matrix{UInt8},min::UInt8,max::UInt8)
    aux = Float32.(frame) .- Float32(min)
    aux ./= Float32(max - min)
    aux .*= 255
    frame[:,:] = converttouint8.(aux)
end

function squeezeframe!(frame::Matrix{Gray{N0f8}},min::Gray{N0f8},max::Gray{N0f8})
    aux = Float32.(frame) .- Float32(min)
    aux ./= Float32(max - min)
    frame[:,:] = Gray.(converttoN0f8.(aux))::Matrix{Gray{N0f8}}
end

function squeezeframe!(frame::Matrix{Float32},min::Float32,max::Float32)
    if max-max<1e-3 return frame end
    aux = frame .- min
    aux ./= (max - min)
    frame[:,:] = aux::Matrix{Float32}
end


function converttouint8(r)
    UInt8(   clamp( round(r), 0, 255 )   )
end

function converttoN0f8(r::Float32)
    N0f8(   clamp( r, 0, 1 )   )
end

Base.convert(::Type{N0f8},a::Number) = N0f8(   clamp( a, 0, 1 )   )
Base.convert(::Type{Gray{N0f8}},a::Real) = N0f8(   clamp( a, 0, 1 )   )



getpixeliterators(N::Int) = collect(Iterators.partition(collect(1:N),N÷Threads.nthreads()))


function smoothglobalillumination!(frames::Matrix{UInt8}, fps::Float64)
    println("removing global illumination spikes...")
    filter = digitalfilter(Lowpass( 3; fs=fps), Butterworth(4))
    Threads.@threads for i in 1:size(frames,1)
        r = filtfilt(filter, Float32.(frames[i,:]))
        frames[i,:] = converttouint8.(r)
    end
end




function smoothbaselinedifference!(frames::Matrix{Gray{N0f8}})
    # calculate moving average background
    # without allocating extra memory
    println("smoothed differential motion...")
    pixeliterators = getpixeliterators(size(frames,1))
    Threads.@threads for pixels in pixeliterators
        b = Float32.(frames[pixels,1])
        d = zeros(Float32,size(b))
        for i in 2:size(frames,2)
            # view current frame (as a column vector)
            col = Float32.(frames[pixels, i])
            # update background history
            b = backgroundrate .* col + (1-backgroundrate) .* b
            # create difference image from background
            col -= b
            # expand for bits, and conserve negative differences
            col = (1 .* col .+ 0.5)
            # smooth difference image
            d = diffimagerate .* col + (1-diffimagerate) .* d
            

            d = clamp.( d, 0f0, 1f0 )
            frames[pixels,i] = Gray{N0f8}.(d)
        end
    end
    return frames::Matrix{Gray{N0f8}}
end



function cumulativebaseline!(dframes::Matrix{Float32}) 
    # reconstruct image from differentials
    # without allocating extra memory
    println("cumulate differential motion to presence...")
    pixeliterators = getpixeliterators(size(dframes,1))
    Threads.@threads for pixels in pixeliterators
        # go backwards
        f = dframes[pixels,end]
        for i in size(dframes,2)-1:-1:1

            # inverse functions (for d and b similarly):
            # d = δ * x + (1-δ) * d
            #         ->    x = (1-(1-δ)) * d / δ = δ * d / δ = d

            # view current frame (as a column vector)
            dcol = dframes[pixels, i]
            dcol .-= 0.5      # center to 0 from image center 0.5
            f = f + dcol     # cumulatively update current frame from previous
            f = clamp.( f, 0f0, 1f0 )
            dframes[pixels,i] = f
        end
    end
    return dframes::Matrix{Float32}
end



# frame-elementary differential and cumulative functions for per frame reconstruction export


function updatedifferencefrombaseline!(d::Vector{Float32}, b::Vector{Float32}, x::Vector{Float32})
    # d is the previous difference
    # b is the previous cumulative background smoothing as baseline
    # x is the current movie frame
    # output will be the changed d, b
    db = similar(d)
    pixeliterators = getpixeliterators(length(d))
    Threads.@threads for pixels in pixeliterators
        b[pixels] = backgroundrate .* x[pixels] + (1f0-backgroundrate) .* b[pixels]
        # create difference image from background
        db[pixels] = x[pixels] - b[pixels]
        # conserve negative differences for image
        db[pixels] .+= 0.5f0
        # smooth the difference image
        d[pixels] = diffimagerate .* db[pixels] + (1f0-diffimagerate) .* d[pixels]
        d[pixels] = clamp.( d[pixels], 0f0, 1f0 )::Vector{Float32}
    end
    return d
end


function updatecumulativedifferencefrombaseline!(x::Vector{Float32}, d::Vector{Float32})
    # x is the previous cumulative
    # d is the current difference
    # output will be the changed x
    pixeliterators = getpixeliterators(length(x))
    Threads.@threads for pixels in pixeliterators
        x[pixels] += d[pixels] .- 0.5f0    # cumulatively update current frame from previous, decentering diff from 0.5
        x[pixels] = clamp.( x[pixels], 0f0, 1f0 )::Vector{Float32}
    end
    return x
end







function getabsolutemotion!(am::Vector{Float32},dx::Vector{Float32},λ::Float32=0.5f0)
    # cumulate with an exponential kernel
    am += (1f0-λ).*abs.(dx.-0.5f0) + λ.*am
    am[:] = clamp.( am[:], 0f0, 1f0 )
    return am::Vector{Float32}
end





function foldimages(F::Matrix,ndisplaycolumns::Int,ndisplayrows::Int)
    v = vcat([F[:,begin+(j-1)*ndisplaycolumns*width:j*ndisplaycolumns*width] for j in 1:ndisplayrows]...)
    v = Gray.(v)
    v::Matrix{Gray{N0f8}}
end



function convertcolortoredgreen!(frame,sub)
    L = channelview(frame[sub...])
    mask = L[3,:,:] .< 0.5

    L[3,:,:] .= 0          # remove the blue

    L[1,.!mask] .= 0       # make the red zero, so that green remains for positive values
    L[2,.!mask] = clamp.( (L[2,.!mask] .- 0.5) .* 2, 0, 1)  # invert the green and expand

    L[2,  mask] .= 0       # make the green zero, so that red remains for negative values
    L[1,  mask] = clamp.(1 .- 2 .* L[1,mask],0,1)  # invert the red and expand


    frame[sub...] = colorview(RGB,L)
end




function convolve(x::Vector{Float64},v::Vector{Float64}; mode=:same)
    nx = length(x)
    nvh = length(v)÷2
    xp = zeros(nx+2*nvh) # zeropadding pre and post domain of x
    xp[nvh:nvh+nx-1] = copy(x)    # avoiding modifying function argument
    cp = [ xp[t-nvh:t+nvh]'*v for t in nvh+1:nvh+nx ]
    if mode==:same
        return cp
    else
        error("Convolution mode: $(mode) not implemented")
    end
end





# image morphology utilities


function erode1d!(h)
    x = copy(h)
    h[1] = minimum([x[1],x[1+1]])
    for i in 2:length(x)-1
        h[i] = minimum([x[i-1],x[i],x[i+1]])
    end
    h[end] = minimum([x[end-1],x[end]])
end

function dilate1d!(h)
    x = copy(h)
    h[1] = minimum([x[1],x[1+1]])
    for i in 2:length(x)-1
        h[i] = maximum([x[i-1],x[i],x[i+1]])
    end
    h[end] = minimum([x[end-1],x[end]])
end


