function movementrepertoire(pixelflow::Matrix{Gray{N0f8}}, exampleframe::Vector{Gray{N0f8}}, timestamps)
    nframes = size(pixelflow,2)
    nbodyparts = length(bodypartboundaries)   # number of body parts, last is the total
    fcontextchange = 11901
    @info "analysing movement repertoire of body parts"
    @info "" bodypartnames

    frames = reshape(pixelflow,height, width, :)
    example = reshape(exampleframe, height, width)
    
    
    
    
    
    # plot regions of interest
    ax = plot(size=(width,height), legend=false, margin=0*Plots.px, axis=false)
    plot!(ax,  example)
    for (B,L) in zip(bodypartboundaries,bodypartnames)
        plot!(ax, [B[3],B[3],B[4],B[4],B[3]], [B[1],B[2],B[2],B[1],B[1]], lw=2, color=:orange)
        annotate!(ax, B[3]+20, B[1]+20, text(L, 20, :orange, :left, :top))
    end
    x1,x2 = xlims(ax)
    y2,y1 = ylims(ax)
    coords = [x2-x1,y2-y1] .* [-0.0,0.95] + [x1,y1]
    annotate!(ax, coords..., text('a', "Helvetica Bold", pointsize=20))
    display(ax)
    # savefig(ax, joinpath(outputpath, filename*"-bodyparts,rois.png"))
    savefig(ax, joinpath("../../publish/journals/journal2020spring/figures/","Supp8a_bodyparts,rois.png"))
    savefig(ax, joinpath("../../publish/journals/journal2020spring/figures/","Supp8a_bodyparts,rois.pdf"))
    
    return



    # absolute mean value of difference image in all regions of interest
    absolutemotion = zeros(Float32,nframes,nbodyparts)
    Threads.@threads for f in 1:nframes
        for b in 1:nbodyparts
            B = bodypartboundaries[b]
            absolutemotion[f,b] = mean(abs.(frames[B[1]:B[2],B[3]:B[4],f] .- 0.5f0))
        end
    end
    


    # plot absolute motions
    axs = plot(layout=(1,2), size=(2*500,1*400),legend=false)
    ax = axs[1]
    plot!(ax, absolutemotion, label=hcat(bodypartnames...))

    ax = axs[2]
    # plot the histogram of the absolute motion for each body part
    histogram!(ax, absolutemotion, label=hcat(bodypartnames...), bins=1000, normed=false)
    xlims!(ax,0,0.01)

    # display(axs)



    # fit a histogra for each bodypart
    # fit a kernel density estimate for each bodypart



    # compare histograms between contexts for each bodypart
    # visual context is 1:fcontextchange frames, audio is fcontextchange+1:end
    # for each bodypart on separate subplot, plot the histogram of the absolute
    # motion for each context
    axs = plot(layout=(nbodyparts÷2,2), size=(2*500*1.2,nbodyparts÷2*400*1.2),legend=false)
    edges = 0:0.0001:0.05
    histograms = []
    for b in 1:nbodyparts
        ax = axs[(b-1)÷2+1,b%2+1]

        hv = fit(Histogram, absolutemotion[1:fcontextchange,b], edges).weights
        ha = fit(Histogram, absolutemotion[fcontextchange+1:end,b], edges).weights

        hvn = normalize( fit(Histogram, absolutemotion[1:fcontextchange,b], edges), mode=:probability ).weights
        han = normalize( fit(Histogram, absolutemotion[fcontextchange+1:end,b], edges), mode=:probability ).weights

        plot!(ax, [hv ha], label=["visual" "audio"], lw=2, color=[:navy :darkgreen], legend=[false,:topright][Int(b==1)+1])
        # xlims!(ax,0,0.05)

        title!(ax, bodypartnames[b])
    end
    # display(axs)




    # prepare for export the absolute motion for each bodypart
    trialsframeindices,_,ntrials, ntimecourse = gettrialsframeindices(timestamps)
    pertrialmotionbodyparts = zeros(ntrials,ntimecourse,nbodyparts)
    for b in 1:nbodyparts
        for i in 1:ntrials
            pertrialmotionbodyparts[i,:,b] = absolutemotion[trialsframeindices[i,:startframe]:trialsframeindices[i,:endframe], b]
        end
    end

    # make header and export to csv
    pertrialmotionbodypartsreshaped = reshape(pertrialmotionbodyparts, ntimecourse*ntrials, nbodyparts)
    pertrialmotionbodypartsreshaped = [ hcat(bodypartnames...); pertrialmotionbodypartsreshaped ]
    writedlm(outputpath*filename*"-absolutemotion,bodyparts"*".csv",  pertrialmotionbodypartsreshaped, ',')



    # calculate and export the distribution (number of trials and timepoints at motion levels) for each bodypart
    bodypartsmotionlevels = [0,0.0043,0.00445,0.00465,0.0048,0.005,0.0053,0.0057,0.0062,0.007,0.008,0.01,0.013,0.018,0.026,0.04,Inf] # detailed
    # bodypartsmotionlevels = [0,0.00445,0.0048,0.005,0.0057,0.007,0.01,0.018,0.04,Inf] # lumped
    nbodypartsmotionlevels = length(bodypartsmotionlevels)-1      # number of motion levels, note range edges is +1
    bodypartsmotionlevelsmidpoints = bodypartsmotionlevels[1:end-1] .+ diff(bodypartsmotionlevels)/2 # shift to centers of bins
    bodypartsmotionlevelsmidpoints[end] = bodypartsmotionlevels[end-1]+bodypartsmotionlevels[end-1]-bodypartsmotionlevels[end-2]
    writedlm(outputpath*filename*"-bodyparts-motionenergy,levels"*".csv",  bodypartsmotionlevelsmidpoints, ',')

    bodypartsmotionlevelstrials = zeros(ntrials,ntimecourse,nbodyparts)
    for b in 1:nbodyparts
        for mx in 1:nbodypartsmotionlevels
            # motion levels
            # count per trial motion level id
            bodypartsmovingleveltrialmask = bodypartsmotionlevels[mx] .<= pertrialmotionbodyparts[:,:,b] .<= bodypartsmotionlevels[mx+1]
            # save current value to parameter grid
            bodypartsmotionlevelstrials[bodypartsmovingleveltrialmask,b] .= mx
        end
        writedlm(outputpath*filename*"-bodyparts,"*bodypartnames[b]*"-motionenergy,levelstrials"*".csv",  bodypartsmotionlevelstrials[:,:,b], ',')
    end
    # check participating number of trials
    h = vcat([count(bodypartsmotionlevelstrials.==mx, dims=1) for mx in 1:nbodypartsmotionlevels]...)
    @info "moving levels of body parts" bodypartnames dropdims(sum(h,dims=2),dims=2)'



end



