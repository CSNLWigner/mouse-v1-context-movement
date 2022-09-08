





function getweights(P, S, chunkP, chunkS, timechunks; singlepass=false)
    # reconstruct the weights of the PCA per original pixel

    Ws = Matrix{Gray{N0f8}}[]   # will hold Weights for each PC

    if ! singlepass
        # inner PC pixel weights
        Wi = (sqrt.(S)' .* P )
        

        for pc in [ [k:k for k in 1:reconstructpc ]; [reconstructpc+1:maxpc]  ]   # all pcs and the residuals
            Wm = zeros(Float32, length(timechunks), height, length(pc)*width)     # collect here the outer chunk weights
            Threads.@threads for chx in 1:length(timechunks)
                
                chunkW = (sqrt.(chunkS[chx][1:chunkdim])' .* chunkP[chx][:,1:chunkdim])
                W = chunkW * Wi[:,pc]       # compound weights from the outer and inner PCA
                
                Wm[chx,:,:] = reshape(W, height, length(pc)*width )       # reshape to image for display
            end
            
            W = dropdims(mean(Wm, dims=1),dims=1)            # get the mean over timechunks

            squeezevideo!(W) # standardize for display
            
            push!(Ws, Gray{N0f8}.(W))      # add to per PC collection
        end

    else
        for pc in [ [k:k for k in 1:reconstructpc ]; [reconstructpc+1:maxpc]  ]   # all pcs and the residuals
            Wm = reshape( (sqrt.(S[pc])' .* P[:,pc] ), height, length(pc)*width )
            squeezevideo!(Wm)
            push!(Ws, Gray{N0f8}.(Wm))
        end
    end

    # println("pc $(reconstructpc+1)..oo")    # residual pcs after display pcs
    # push!(X̂s,   broadcast(+,   broadcast(*, P[:,reconstructpc+1:end]*Z[reconstructpc+1:end,:], σ),    μ)   )

    # @info "weights" length(Ws) size(Ws[1])

    return Ws::Vector{Matrix{Gray{N0f8}}}

end





function reconstructframefrompcs(Z, P, μ, σ, chunkP, chunkμ, chunkσ, timechunks, i::Int; reconstructpc=reconstructpc, singlepass=false)
    # reconstruct from PCs, at timeframe i
    
    npixels = size(chunkP[begin],1)

    dx̂s = Vector{Float32}[]    # holds reconstruction list for differential motion

    # go through the pcs and the last extra is for the residuals
    for pc in [ [k:k for k in 1:reconstructpc ]; [reconstructpc+1:maxpc]  ]         # unitranges for single PCs are needed for proper matrix product dot product
        # print("pc", pc, ", ")

        dx̂ = zeros(Float32,npixels)        # allocate the huge memory block for the differential reconstruction
    
        if ! singlepass
            # first reconstruct all chunk from the outer PCA as a single differential Y:

            dŷ = broadcast(+,   broadcast(*, P[:,pc]*Z[pc,i], σ),   μ)

            # then reconstruct based on which chunk the frame belongs to, from dy to original chunk pixel size from inner PCAs for each chunk
            for (chx,timechunk) in enumerate(timechunks)
                # print("$(timechunk[begin]):$(timechunk[end]), ")
                if i in timechunk
                    dx̂ = dropdims(broadcast(+,   broadcast(*, chunkP[chx]*dŷ, chunkσ[chx]),   chunkμ[chx]),dims=2)
                end
            end
        else      # in case there is no multiple chunks, only the inner PCA is needed
            dx̂ = dropdims(broadcast(+,   broadcast(*, P[:,pc]*Z[pc,i], σ),   μ)[:,1],dims=2)
        end

        push!(dx̂s, dx̂)
    end

    # println()


    # println("pc $(reconstructpc+1)..oo")    # residual pcs after display pcs
    # push!(X̂s,   broadcast(+,   broadcast(*, P[:,reconstructpc+1:end]*Z[reconstructpc+1:end,:], σ),    μ)   )

    # @info "reconstruction" length(dx̂s) size(dx̂s[1])

    return dx̂s
end
















function reconstruct(X,Z,P,S,μ,σ,chunkZ,chunkP,chunkS,chunkμ,chunkσ,timechunks,fps,fa; singlepass=false, writevideo=false, writeimage=true)
    # open IO (video and selected frame images)
    # reconstruct frame by frame and export
    # @info "outer" P S mean(P) std(P) length(chunkP)
    # @info "inner" chunkP chunkS [mean(cP) for cP in chunkP] [std(cP) for cP in chunkP]
    # display(histogram(P[:]))
    # display(plot(layout=(2,3),[histogram(cP[:]) for cP in chunkP]))
    
    # select exporting frames, videoindices must be contiguous due to the differential-cumulative operation
    exportvideoframeindices = exportframestart:exportframestart+exportframeduration-1
    # exportvideoframeindices = 9501:9500+4800
    # exportvideoframeindices = 1:10

    exportimageframeindices = []
    # exportimageframeindices = exportframestart:exportframestart+exportframeduration-1
    exportimageframeindices = 6848+8:6848+13      # a paw movement
    # exportimageframeindices = [18,19,20,98,99,100,198,199,200,298,299,300]
    # exportimageframeindices = 1:10#16:20

    
    # display folding
    ndisplayrows = 2
    ndisplaycolumns = reconstructpc÷ndisplayrows
    hdiv = 5   # divisor for the PC height

    # open writer object
    framesize = (length(1:downsample:(height+(height÷hdiv+3*height)*ndisplayrows)), length(1:downsample:(width*ndisplaycolumns)))
    if writevideo
        fnpart = "-pc-reconstruction"     # dvc.yaml default export
        # fnpart = "-pc-reconstruction-f$(lpad(exportframestart,5,'0')),$exportframeduration,d$downsample"        # diagnostic filenames
        io = open_video_out(joinpath(outputpath,filename*fnpart*ext),
                            RGB{N0f8}, framesize, framerate=fps,
                            codec_name="libx264rgb",encoder_options=(crf=0,preset="medium"))
        #
    end
    @info "frame export" writevideo writeimage framesize


    # one pad for the annotations, together two less from reconstructpc number for the rest after the original and diff original
    padding = zeros(Gray{N0f8},height,width)


    # get the filter weights, averaging over chunks
    Wm = hcat( getweights(P, S, chunkP, chunkS, timechunks; singlepass=singlepass)[1:reconstructpc]... )     # use only the main pc Wms but not the residuals

    # state maintenance b,d,x̂s
    # initialize the first smoothed baseline to the first movie frame
    b = Float32.(X[:,exportvideoframeindices[begin]])
    # initialize the first differential original to the first movie frame
    d = Float32.(X[:,exportvideoframeindices[begin]]) - b .+ 0.5f0
    # initialize the first reconstruction to be the first frame differential decentered (reconstructions will gradually build up from the first frame and on)
    dx̂s = reconstructframefrompcs(Z, P, μ, σ, chunkP, chunkμ, chunkσ, timechunks, exportvideoframeindices[begin]; singlepass=singlepass)
    x̂s = deepcopy(dx̂s)         # x̂s and cx̂s record the frame-cumulating image pixel intensities adding each diff frame dx̂s
    x̂s = [ updatecumulativedifferencefrombaseline!( x̂s[pc], dx̂s[pc].-0.5f0 ) for pc in eachindex(x̂s) ]
    cx̂s = deepcopy(x̂s)

    # PC statistics
    # pcmeans = dropdims(mean(Z,dims=2),dims=2)
    # pcstds = dropdims(std(Z,dims=2),dims=2)

    
    
    # iterate state updates and write requested frames
    print("processing frame ")
    for i in exportvideoframeindices
        if i%10==0 print("$i, ") end

        
        
        # collect the parts of the frame

        
        # original movie frame
        x = reshape(X[:,i],height,width)


        
        # calculate the original differential motion image
        updatedifferencefrombaseline!(d, b, Float32.(X[:,i]))
        dx = copy(reshape(d,height,width))
        

        
        # pcs (will be shown as small uniform color rectangles)
        pcs = hcat([  repeat([Z[px,i]],height÷hdiv, width) for px in 1:reconstructpc ]...)
        standardizevideo!(pcs)
        pcs = convert.(Gray{N0f8}, pcs)






        # reconstruction: differential reconstruction
        dx̂s = reconstructframefrompcs(Z, P, μ, σ, chunkP, chunkμ, chunkσ, timechunks, i; singlepass=false)

        # cumulate over frames differential for original reconstruction
        x̂s = [ updatecumulativedifferencefrombaseline!(x̂s[pc],dx̂s[pc]) for pc in eachindex(x̂s) ]


        # reconstruct from increasingly more PCs of dx̂s and cumulate them, removing the 0.5 centering, and in the end add it back
        cdx̂s = [ zeros(Float32, height*width) ]
        for pc in 1:length(dx̂s)
            push!(cdx̂s, cdx̂s[end]+dx̂s[pc].-0.5f0)
        end
        deleteat!(cdx̂s,1)
        map!(v->v.+0.5f0,cdx̂s, cdx̂s)      # add back 0.5 centering after the cumulative pc and frame to cx̂s updates
        cx̂s = [ updatecumulativedifferencefrombaseline!(cx̂s[pc],cdx̂s[pc]) for pc in eachindex(cdx̂s) ]



        # # make the differential visible (individual and pc-cumulative)
        # standardizevideo!(dx, shift=0.5f0, σ=6f0)

        # standardizevideo!.(dx̂s, shift=0.5f0, σ=6f0)
        # standardizevideo!.(cdx̂s, shift=0.5f0, σ=6f0)
        # # make the time-cumulative reconstruction visible (individual and pc-cumulative)
        # standardizevideo!.(x̂s, shift=0.2f0, σ=6f0)
        # standardizevideo!.(cx̂s, shift=0.2f0, σ=6f0)

        # make the differential visible (individual and pc-cumulative)
        print("dx")
        standardizevideo!(dx, shift=0.5f0, σ=2f0, absolute=true, shows=false)

        print("dx̂s")
        standardizevideo!.(dx̂s, shift=0.5f0, σ=10f0, absolute=true, shows=false)
        print("cdx̂s")
        standardizevideo!.(cdx̂s, shift=0.5f0, σ=3f0, absolute=true, shows=false)
        # make the time-cumulative reconstruction visible (individual and pc-cumulative)
        standardizevideo!.(x̂s, shift=0.2f0, σ=6f0)
        standardizevideo!.(cx̂s, shift=0.2f0, σ=6f0)



        # convert to image format
        dxspcs = convert.(Gray{N0f8},dx)

        dx̂spcs = convert.(Gray{N0f8}, reshape(hcat(dx̂s[1:reconstructpc]...), height, :))
        x̂spcs = convert.(Gray{N0f8}, reshape(hcat(x̂s[1:reconstructpc]...), height, :))
        
        # register the residual (reconstructpc+1:end) differential and cumulative reconstructions, 
        # it was calculated cumulatively for all in the pca step, stored at the reconstructpc+1 index (hence "end")
        rdx̂spcs = convert.(Gray{N0f8}, reshape(hcat(dx̂s[end]...), height, :))
        rx̂spcs = convert.(Gray{N0f8}, reshape(hcat(x̂s[end]...), height, :))

        
        # register the differential and cumulative reconstructions from all PCs, which is the cumulative sum until the last dx̂s and x̂s entry:
        tcdx̂spcs = convert.(Gray{N0f8}, reshape(hcat(cdx̂s[end]...), height, :))
        tcx̂spcs = convert.(Gray{N0f8}, reshape(hcat(cx̂s[end]...), height, :))






        # put together the frame from parts

        # original, differential, padding
        # loadings of individual principal components in the image
        # pc multipliers (small height)
        # differential reconstructions from individual principal components
        # reconstructions of the original cumulating the differential reconstruction in frame times
        frame = [ [ x;; dxspcs;;  padding;; repeat(padding,1,ndisplaycolumns-3-4);; rdx̂spcs;; rx̂spcs;; tcdx̂spcs;; tcx̂spcs  ];
                    foldimages( [Wm; pcs; dx̂spcs; x̂spcs ], ndisplaycolumns, ndisplayrows )
                      ]
        #


        # triplicate for color image
        frame = RGB.(frame)

        # replace loadings and PCs with color values
        subdiff = [1:1*height, 1*width+1:2*width]
        subrdiff = [1:1*height, (ndisplaycolumns-4)*width+1:(ndisplaycolumns-3)*width]
        subtdiff = [1:1*height, (ndisplaycolumns-2)*width+1:(ndisplaycolumns-1)*width]
        # subloadings = [1*height+1:2*height, :]
        # subweights = [2*height+1:2*height+height÷hdiv, :]
        # subdreconstructs = [2*height+height÷hdiv+1:3*height+height÷hdiv, :]
        # subdreconstructscumulative = [4*height+height÷hdiv+1:5*height+height÷hdiv, :]

        subdiffreconstr = [ vcat([collect(1*height+1:1*height+(2*height+height÷hdiv)).+(k-1)*(3*height+height÷hdiv) for k in 1:ndisplayrows ]...)   , :]

        for sub in [subdiff,subrdiff,subtdiff,subdiffreconstr]
            convertcolortoredgreen!(frame,sub)
        end



        # downsample for export and display
        eframe = frame[begin:downsample:end,begin:downsample:end]

        # add annotation to the empty padding with fixed resolution, independent of downsampling
        if (!ismissing(fa[i,:degree]) && fa[i,:degree]!=-1) || (!ismissing(fa[i,:freq]) && fa[i,:freq]!=-1)
            annotation = annotationoverlaybitmap(fa[i,:degree], fa[i,:freq], fa[i,:block])
            annotation = colorview(RGB{N0f8}, permutedims(repeat(annotation,1,1,3),(3,1,2)))
            eframe[A_heightoffsetstart:A_heightoffsetend,A_widthoffsetstart+width÷downsample*2:A_widthoffsetend+width÷downsample*2] = annotation
        end



        # export video frame to stream
        if writevideo
            write(io, eframe)
        end

        # export still image for selected frames
        if i in exportimageframeindices
            # ax = plot(eframe,size=(4*reconstructpc*200,4*(6*200+200÷5)))
            # display(ax)
            # if writeimage
            #     savefig(ax, joinpath(outputpath,"reconstructionframes",filename*"-pc-reconstruction-f$(lpad(i,5,'0')).png"))
            # end
            iframe = [ frame[1:height, 1:1*width];; frame[1:height, 1*width+1:2*width];; frame[1:height, 5*width+1:6*width] ]
            ax = plot(iframe,size=(width*3,height))
            display(ax)
            if writeimage
                # savefig(ax, joinpath(outputpath,"reconstructionframes",filename*"-pc-reconstruction-small-f$(lpad(i,5,'0')).png"))
                save(joinpath(outputpath,"reconstructionframes",filename*"-pc-reconstruction-small-f$(lpad(i,5,'0')).png"), iframe)

            end
        end



    end
    println()


    # close the video stream
    if writevideo
        close_video_out!(io)
    end


end












function diagnosetrialboundaries(timestamps, fa)

    pad = [-1.5 1.5]
    # pad = [0 0]
    trialsframeindices, events = gettrialboundaryframes(timestamps; paddingstart=pad[1], paddingend=pad[2])
    
    # trialsframeindices events

    D = [ trialsframeindices[:,:startframetime] .- pad[1];; trialsframeindices[:,:endframetime] .- pad[2] ;; events[:,:start];; events[:,:start]+events[:,:duration] ]
    append!(fa,fa[end-100:end,:])
    @info "fa" fa
    offset = 29
    D = [ D;; fa[trialsframeindices[:,:startframe].+offset,:timestamp];; fa[trialsframeindices[:,:startframe].+offset,:degree];; fa[trialsframeindices[:,:endframe].+offset,:freq] ]

    
    @info "t f ind" columns=["frame start","frame end", "events start", "events end","annotation start", "annotation values..."] D



    return
end





function viewstationarytrials(X::Matrix{Gray{N0f8}}, movingtrial::BitVector, timestamps, fps, fa, writevideo=true, writeimage=false)

    timepad = 1.5
    trialsframeindices, events = gettrialboundaryframes(timestamps; paddingstart=-timepad, paddingend=timepad)
    stationarytrialsframedata = trialsframeindices[.!movingtrial,:]
    stationaryevents = events[.!movingtrial,:]

    stationarytrialframesindices = vcat([ collect(e[:startframe]:e[:endframe]) for e in eachrow(stationarytrialsframedata) ]...)

    @info "trials" size(stationarytrialframesindices) typeof(stationarytrialframesindices) stationarytrialframesindices


    exportvideoframeindices = stationarytrialframesindices

    exportimageframeindices = []
    # exportimageframeindices = exportvideoframeindices


    wdiv = 5

    # open writer object
    framesize = (length(1:downsample:height), length(1:downsample:(width+width+width÷wdiv))+1)
    @info "frame export" writevideo writeimage framesize
    fnpart = "-motionenergy,threshold-view"     # dvc.yaml default export
    if writevideo
        io = open_video_out(joinpath(outputpath,filename*fnpart*ext),
                            RGB{N0f8}, framesize, framerate=fps,
                            codec_name="libx264rgb",encoder_options=(crf=0,preset="medium"))
        #
    end


    # one pad for the annotations, together two less from reconstructpc number for the rest after the original and diff original
    padding = zeros(Gray{N0f8},height,width÷wdiv+1)

    # initialize the first smoothed baseline to the first movie frame
    b = Float32.(X[:,exportvideoframeindices[begin]])
    # initialize the first differential original to the first movie frame
    d = Float32.(X[:,exportvideoframeindices[begin]]) - b .+ 0.5f0
    # initialize the first reconstruction to be the first frame differential decentered (reconstructions will gradually build up from the first frame and on)

    # iterate state updates and write requested frames
    print("processing frame ")
    jx = 0
    for i in exportvideoframeindices

        # if i < exportvideoframeindices[35] || i > exportvideoframeindices[38] continue end
        jx += 1
        if jx%10==0 print("$i, ") end


        # collect the parts of the frame

        
        # original movie frame
        x = reshape(X[:,i],height,width)


        
        # calculate the original differential motion image
        updatedifferencefrombaseline!(d, b, Float32.(X[:,i]))
        dx = copy(reshape(d,height,width))
        

        


        # make the differential visible (individual and pc-cumulative)
        standardizevideo!(dx, shift=0.5f0, σ=6f0)

        # convert to image format
        dxspcs = convert.(Gray{N0f8},dx)





        # put together the frame from parts

        # original, differential, padding
        # loadings of individual principal components in the image
        # pc multipliers (small height)
        # differential reconstructions from individual principal components
        # reconstructions of the original cumulating the differential reconstruction in frame times
        frame = [ x;; dxspcs;;  padding  ]
        #


        # triplicate for color image
        frame = RGB.(frame)

        # replace differential with color values
        subdiff = [1:1*height, 1*width+1:2*width]
        convertcolortoredgreen!(frame,subdiff)
        

        # downsample for export and display
        eframe = frame[begin:downsample:end,begin:downsample:end]

        # add annotation to the empty padding with fixed resolution, independent of downsampling
        if (!ismissing(fa[i,:degree]) && fa[i,:degree]!=-1) || (!ismissing(fa[i,:freq]) && fa[i,:freq]!=-1)
            # @info "trial" fa[i,:]
            annotation = annotationoverlaybitmap(fa[i,:degree], fa[i,:freq], fa[i,:block])
            annotation = colorview(RGB{N0f8}, permutedims(repeat(annotation,1,1,3),(3,1,2)))
            eframe[A_heightoffsetstart:A_heightoffsetend,A_widthoffsetstart+width÷downsample*2:A_widthoffsetend+width÷downsample*2] = annotation
        end



        # export video frame to stream
        if writevideo
            write(io, eframe)
        end

        # export still image for selected frames
        if i in exportimageframeindices
            ax = plot(eframe,size=(200+200+200÷5, 200))
            # display(ax)
            if writeimage
                savefig(ax, joinpath(outputpath,"reconstructionframes",filename*fnpart*"-f$(lpad(i,5,'0')).png"))
            end
        end



    end
    println()


    # close the video stream
    if writevideo
        close_video_out!(io)
    end

end