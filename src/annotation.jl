# Annotation bitmaps
const A_v45 = [ 0 0 0 0 0 0 1;
               0 0 0 0 0 1 1;
               0 0 1 1 1 1 1;
               1 1 1 1 1 1 1;
               0 0 1 1 1 1 1;
               0 0 0 0 1 1 1;
               0 0 0 0 0 0 1  ]
const A_v135 = reverse(A_v45, dims=2)
const A_a5000 = permutedims(A_v135)
const A_a10000 = reverse(A_a5000,dims=1)
const A_cxv = [ 1 0 0 0 0 0 1;
                1 0 0 0 0 0 1;
                0 1 0 0 0 1 0;
                0 1 0 0 0 1 0;
                0 0 1 0 1 0 0;
                0 0 1 0 1 0 0;
                0 0 0 1 0 0 0  ]
const A_cxa = [ 0 0 0 1 0 0 0;
                0 0 1 0 1 0 0;
                0 0 1 0 1 0 0;
                0 1 0 0 0 1 0;
                0 1 1 1 1 1 0;
                1 0 0 0 0 0 1;
                1 0 0 0 0 0 1  ]
const A_heightoffsetstart = 11
const A_widthoffsetstart = 11
const A_heightoffsetend = A_heightoffsetstart - 1 + size(A_v45,1)
const A_widthoffsetend = A_widthoffsetstart - 1 + size(A_v45,1)*5



function gettrialboundaryframes(timestamps; paddingstart=0, paddingend=0)
    events = loadevents()
    deleteat!(events, events[!,:block] .% 2 .== 1)
    ncomplextrials = nrow(events)
    trialsframeindices = zeros(Int, ncomplextrials, 2)
    trialon = false
    lasttrial = 0
    for (f,t) in enumerate(timestamps)      # f is frame number, t is time in seconds
        i = findfirst((t .> events[!,:start] .+ paddingstart) .& (t .< events[!,:start] .+ 3. .+ paddingend ))
        if isnothing(i)
            if trialon
                trialsframeindices[lasttrial,2] = f # add the trial end timeframe
            end
            trialon = false
        else
            if ! trialon
                trialsframeindices[i,1] = f    # add the trial start timeframe
                trialon = true
            end
            lasttrial = i
        end
    end

    trialsframeindices = DataFrame(:startframe=>trialsframeindices[:,1], :endframe=>trialsframeindices[:,2],
                                   :startframetime=>timestamps[trialsframeindices[:,1]], :endframetime=>timestamps[trialsframeindices[:,2]] )
    return trialsframeindices, events
end




function annotateframes(timestamps)
    events = loadevents()
    fa = DataFrame(timestamp=Float32[], duration=Float32[], block=Int[], degree=Union{Float32,Missing}[], freq=Union{Float32,Missing}[])
    for t in timestamps
        i = findfirst((t .> events[!,:start] ) .& (t .< events[!,:start] .+ 3.))
        if isnothing(i)
            push!(fa,[ Float32(t) Float32[-1 -1 -1 -1] ] )
        else
            e = events[i,:]
            push!(fa,[ Float32(t) Vector(e[2:5])' ] )
        end
    end
    return fa
end



function annotationoverlaybitmap(visual::Union{Float32,Missing}, audio::Union{Float32,Missing}, context::Int)
    zeropad = zeros(Gray{N0f8}, size(A_v45))
    if ismissing(visual) visualbitmap = zeropad else visualbitmap = [A_v45,A_v135][Int(visual==135)+1] end
    if ismissing(audio) audiobitmap = zeropad else audiobitmap = [A_a5000,A_a10000][Int(audio==10000)+1] end
    contextbitmap = [A_cxv, A_cxa][Int(context>=3)+1]
    bitmap = hcat( visualbitmap, zeropad, audiobitmap, zeropad, contextbitmap )
    return Gray{N0f8}.(bitmap)::Matrix{Gray{N0f8}}
end

