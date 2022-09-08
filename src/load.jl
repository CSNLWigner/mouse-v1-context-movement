function loadevents()
    eventfilename = rawbehaviourrootpath*"$(mouseid)/$(mouseid)-behaviour.csv"
    events = DataFrame(CSV.File(eventfilename, missingstring=[""]))[!,[:start,:duration,:block,:degree,:freq,:punish]]
    events[!,[:start,:duration]] ./= 1000.
    events[!,:block] .+= 1.
    return events
end







function getvideoinfo(path::String)
    @info "getvideoinfo" path
    numberofframes = VideoIO.get_number_frames(path)
    duration = VideoIO.get_duration(path)
    fps = Int(round(numberofframes/duration))
    return numberofframes, duration, fps
end



function loadvideo(path::String, crop::Vector{Int})
    println("loading video...")
    numberofframes, duration, fps = getvideoinfo(path)
    timestamps = 0
    if haskey(video, :align)
        align = DataFrame( CSV.File(joinpath(inputpath,video[:align]*"-synchronizedtimestamps.csv")) )
        timestamps = align.timestamps[framestart:framestart+frameduration-1]
        fps = round( 1/mean(diff(timestamps)), digits=1 )
    else
        timestamps = collect(0:1/fps:(frameduration-1.)/fps) .+ framestart/fps
    end
    width,height = (crop[2]-crop[1]), (crop[4]-crop[3])
    inputdims = width*height


    @info "params" mouseid filename numberofframes duration fps framestart frameduration+framestart frameduration framestart/fps (frameduration+framestart)/fps frameduration/fps crop'
    
    frames = zeros(UInt8,inputdims,frameduration)
    vf = openvideo(path, target_format=VideoIO.AV_PIX_FMT_GRAY8, thread_count=min(Threads.nthreads(),4))
    for (i,f) in enumerate(vf)
        if i<framestart continue end
        if i>=frameduration+framestart break end
        if mod(i,100)==0 print("$(i) ") end

        # ff = Float32.(reinterpret(UInt8,f))./255
        ff = reinterpret(UInt8,f)
        cff = ff[crop[3]+1:crop[4],crop[1]+1:crop[2]]          # array and image dimensions are in opposite order
        rcff = reshape(cff,inputdims)
        frames[:,i-framestart+1] = rcff

    end
    println("(last)")
    close(vf)

    # ax = plot(frames[5])
    # display(ax)

    # @info "frames" frames

    return fps, timestamps, frames
end





function concatenateblocks()
    timestamps = []
    X = []
    global fps
    for videofilename in session[:videos]
        (fps,timestampspart,Xpart) = deserialize(joinpath(outputpath,videofilename*"-cropped-raw"*".dat"))
        push!(timestamps, timestampspart)
        push!(X, Xpart)
    end
    timestamps = vcat(timestamps...)
    X = hcat(X...)
    @info "concatentation" size(timestamps) size(X)
    return fps, timestamps, X
end