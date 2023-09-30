

function pcatransform(frames::Matrix{Gray{N0f8}}; singlepass=false)
    if singlepass println("performing pca with one pass...") else println("performing pca with chunkσ...") end

    
    
    # process PCA w/SVD in disjoint time parts (chunkσ) of the video
    timechunks = collect(Iterators.partition(collect(1:size(frames,2)),chunksize))
    @info "chunks" chunkdim timechunks

    chunkμ = Matrix{Float32}[] # this will hold the means of the features in chunkσ
    chunkσ = Matrix{Float32}[] # this will hold the std of the features in chunkσ
    chunkP = Matrix{Float32}[] # this will hold the projection matrices for each calculated PCA chunk
    chunkS = Vector{Float32}[] # this will hold the singular values for each calculated PCA chunk as a vector
    chunkZ = Matrix{Float32}(undef,chunkdim,0)  # this will hold separately calculated PCs of each chunk concatenated in time
    for timechunk in timechunks
        print("$(timechunk[begin]):$(timechunk[end]), ")
        X = Float32.(frames[:,timechunk])

        # center
        μ = mean(X,dims=2)
        X .-= μ
        push!(chunkμ, μ)

        # standardize
        σ = std(X,dims=2)
        maskwithnovar = σ.<0.0001          # exclude close to zero time-variance pixels from standardization
        σ[maskwithnovar] .= 1.
        X ./= σ
        # display(histogram(σ))
        push!(chunkσ, σ)
        

        # use the multivariatestats pca method, without calculated noise-proportion dimension reduction:
        pca = fit(PCA, X, mean=0, maxoutdim=chunkdim, pratio=1.0)
        P = projection(pca)
        push!(chunkP, P)
        @info "outer PCAs" principalvars(pca)[1:10]' principalvars(pca)[1:10]'./tprincipalvar(pca) tprincipalvar(pca)/(tprincipalvar(pca)+tresidualvar(pca))

        S = principalvars(pca)
        push!(chunkS,S)

        Z = P'*X      # transform into PCA space
        chunkZ = [chunkZ Z ]
        
    end
    println("")
    # @info "chunks size" size(chunkZ)
    


    
    if ! singlepass
        # now with the reduced video dimensions for each chunks concatenated in time, do the final PCA

        # center
        μ = mean(chunkZ,dims=2)
        chunkZ .-= μ
        
        # standardize
        σ = std(chunkZ,dims=2)
        maskwithnovar = σ.<0.0001          # exclude close to zero time-variance pixels from standardization
        σ[maskwithnovar] .= 1.
        chunkZ ./= σ


        pca = fit(PCA, chunkZ, mean=0, maxoutdim=maxpc)#, pratio=0.9)
        P = projection(pca)
        S = principalvars(pca)
        Z = P'*chunkZ

        @info "sizes" size(frames) size(P) size(Z)


        @info "inner PCA" principalvars(pca)' tprincipalvar(pca) tresidualvar(pca) tvar(pca)
        @info "projection dispersion" mean(P,dims=1) std(P,dims=1)

    else     # if single pass, only use inner PCA as outer PCA

        P = chunkP[1]
        S = chunkS[1]
        Z = chunkZ
        μ = chunkμ[1]
        σ = chunkσ[1]

    end
    return Z, P, S, μ, σ, chunkZ, chunkP, chunkS, chunkμ, chunkσ, timechunks
end




function exportpctimeseries(Z,timestamps, fileid::String)
    println("exporting principal component projections...")
    header = ["time" ["$(fileid)$(i)" for i in 1:size(Z,1)]... ]
    @info "sizes" size(header) size(timestamps) size(Z')
    data = [ header; timestamps Z' ]
    writedlm(outputpath*filename*"-$(fileid)-latent"*".csv",  data, ',')


    # CSV()
end




function displaypctimeseries(Z,timestamps, fileid::String)
    println("exporting principal component projections...")
    
    n_timestamps = size(Z,2)

    zoomlabels = ["long","short"]
    zoomindices = [1:n_timestamps, exportframestart:exportframestart+exportframeduration-1]
    kernel = ones(Float32,kernelhalfwidth*2+1)./(kernelhalfwidth*2+1)
    z = similar(Z)

    for (label,zoomindex) in zip(zoomlabels,zoomindices)
        ax = plot(layout=(reconstructpc,1),size=(2400, reconstructpc*200), legend=false)
        for px in 1:reconstructpc

            z[px,zoomindex] = conv( abs.(Z[px,zoomindex]), kernel )[kernelhalfwidth+1:end-kernelhalfwidth]

            axs = ax[px]
            plot!(axs, timestamps[zoomindex], Z[px,zoomindex], color=:grey, lw=1, ylabel="PC $px")
            plot!(axs, timestamps[zoomindex], z[px,zoomindex], color=:gold, lw=1, ylabel="PC $px abs smooth")
            hline!(axs,[0],color=:darkgrey,lw=0.5,alpha=0.5)

        end
        display(ax)
        savefig(ax, joinpath(outputpath,filename*"-$(fileid)-latent-$label.png"))
    end

end






function calculateabsolutemotion(Z)
    npcs, ntimestamps = size(Z)
    E = zeros(npcs,ntimestamps)

    kernel = ones(Float32,kernelhalfwidth*2+1)./(kernelhalfwidth*2+1)

    # calculate linear approximation of smoothed motion energy, and their stats over principal components as z,s
    for pc in 1:npcs
        E[pc,:] = conv( abs.(Z[pc,:]), kernel )[kernelhalfwidth+1:end-kernelhalfwidth]
    end
    z = mean(E,dims=1)
    s = std(E,dims=1)

    return E,z,s
end



function gettrialsframeindices(timestamps)
    trialsframeindices, events = gettrialboundaryframes(timestamps, paddingstart=-1.5, paddingend=1.5)
    @info "t" trialsframeindices
    # correct some frame inprecision
    ntimecourse = 121
    trialwidths = trialsframeindices[!,:endframe]-trialsframeindices[!,:startframe]
    @info "tneq" trialwidths[trialwidths.!=ntimecourse-1] 
    @info "trialwidths pre correction" unique(trialwidths)

    trialsframeindices[trialwidths.==121,:endframe] .-= 1
    trialsframeindices[trialwidths.==119,:endframe] .+= 1
    trialsframeindices[trialwidths.==111,:startframe] .-= 9

    trialwidths = trialsframeindices[!,:endframe]-trialsframeindices[!,:startframe]
    @info "trialwidths post correction" unique(trialwidths)

    ntrials = nrow(trialsframeindices)

    return trialsframeindices, events, ntrials, ntimecourse
end




function threshold(fps,timestamps,Z,P,S,μ,σ,chunkZ,chunkP,chunkS,chunkμ,chunkσ,timechunks)
    
    _,z,s = calculateabsolutemotion(Z)


    # calculate a per trial statistics as well
    trialsframeindices, events, ntrials, ntimecourse = gettrialsframeindices(timestamps)

    # collect stats and heatmap data
    pertrialstats = zeros(ntrials,2)
    pertrialmotion = zeros(ntrials,ntimecourse)
    for i in 1:ntrials
        pertrialstats[i,:] = [mean(z), mean(s)]
        pertrialmotion[i,:] = z[trialsframeindices[i,:startframe]:trialsframeindices[i,:endframe]]
    end
    @info "pertrial" size(pertrialmotion)
    
    # define display parameters    
    motionthresholddisplay = 0.4
    proportiontrialmotionthresholddisplay = 0.1
    movingtrialdisplay = nothing
    nmotiontrialsdisplay = nothing
    
    
    nmotionthresholds = length(motionthresholds)
    # first column: proportion value, rest motiontrials BitVector
    motionthresholdstrials = zeros(length(proportiontrialmotionthresholds)*ntrials, 1+nmotionthresholds)
    

    # get threshold parameter grid triallists and pack them for export to csv
    for (mx,motionthreshold) in enumerate(motionthresholds)
        for (px,proportiontrialmotionthreshold) in enumerate(proportiontrialmotionthresholds)

            # stationarity
            # create per trial threshold mask
            movingtrial = dropdims(mean(pertrialmotion .>= motionthreshold,dims=2),dims=2) .> proportiontrialmotionthreshold
            # save current value to parameter grid
            motionthresholdstrials[(px-1)*ntrials+1:px*ntrials,1] .= proportiontrialmotionthreshold
            motionthresholdstrials[(px-1)*ntrials+1:px*ntrials,1+mx] = Float64.(movingtrial)




            # store display variables
            if motionthreshold==motionthresholddisplay && proportiontrialmotionthreshold==proportiontrialmotionthresholddisplay
                movingtrialdisplay = copy(movingtrial)
                nmotiontrialsdisplay = sum(movingtrial)
                # cross check if no funny businness is going on with the stationary trials
                @info "characteristics of stationary trials" events[.!movingtrialdisplay,:]
            end

        end


    end



    # get levels

    motionlevels = [0.0; motionthresholds[1:end-1]; Inf]
    # find the best bin edges that give equal numbers of trials in each bin

    motionlevels = [0,collect(0.06:0.04:0.32)...,0.4,0.6,1.0,1.4,2,Inf]

    nmotionlevels = length(motionlevels)-1      # number of motion levels, note range edges is +1
    # first column: proportion value, rest motiontrials levels id
    motionlevelstrials = zeros(ntrials,ntimecourse)

    for mx in 1:nmotionlevels
        # motion levels
        # count per trial motion level id
        movingleveltrialmask = motionlevels[mx] .<= pertrialmotion .<= motionlevels[mx+1]
        # save current value to parameter grid
        motionlevelstrials[movingleveltrialmask] .= mx
    end

    h = vcat([count(motionlevelstrials.==mx, dims=1) for mx in 1:nmotionlevels]...)
    @info "moving levels" motionlevelstrials h sum(h,dims=2)
    motionlevelsmidpoints = motionlevels[1:end-1] .+ diff(motionlevels)/2 # shift to centers of bins



    # figure



    excludelevel = -0.5   # this is the default color palette code for trials not included in the given set of threshold combinations
    pertrialmotionmoving = excludelevel .* ones(size(pertrialmotion))
    pertrialmotionstationary = excludelevel .* ones(size(pertrialmotion))
    pertrialmotionmoving[movingtrialdisplay,:] = pertrialmotion[movingtrialdisplay,:]
    pertrialmotionstationary[.!movingtrialdisplay,:] = pertrialmotion[.!movingtrialdisplay,:]

    # calculate statistics to determine if the threshold is good enough
    movingdisplay = z .>= motionthresholddisplay
    @info "moving" numframesmoving = sum(movingdisplay) percentframesmoving = sum(movingdisplay)/length(movingdisplay)
    @info "ts" timestamps[1:5] timestamps[end-4:end]

    @info "motion energy" size(z)

    trialframetimestamps = collect(-1.5:0.05:4.5)
    bins = 0:0.05:5
    clims = (excludelevel,2)


    ax = plot(layout=(3,3),size=(3*500,3*400), top_margin=40Plots.px, bottom_margin=20Plots.px,
                                               left_margin=30Plots.px, right_margin=30Plots.px, legend=false )
    

    axs = ax[1,1]
    histogram!(axs, z[.!movingdisplay], bins=bins, color=:blue, label="stationary")
    histogram!(axs, z[movingdisplay], bins=bins, color=:red, label="moving")
    plot!(axs, [motionthresholddisplay,motionthresholddisplay], [0,4000], ls=:dash, color=:grey, label="threshold", legend=true)
    xlims!(axs,0,3)
    xlabel!(axs, "mean over PCs")
    ylabel!(axs, "number of frames")

    axs = ax[1,2]
    histogram!(axs, s[.!movingdisplay], bins=bins, color=:blue)
    histogram!(axs, s[movingdisplay], bins=bins, color=:red)
    xlims!(axs,0,3)
    xlabel!(axs,"std over PCs")
    # title!(axs,"Absolute Motion\n\nsmooth absolute motion mean over pcs < stationary threshold: $motionthresholddisplay")
    title!(axs,"Smoothed absolute motion")

    axs = ax[1,3]
    scatter!(axs, z[.!movingdisplay], s[.!movingdisplay], color=:blue)
    scatter!(axs, z[movingdisplay], s[movingdisplay], color=:red)
    xlabel!(axs,"mean over PCs")
    ylabel!(axs,"std over PCs")



    
    axs = ax[2,1]
    heatmap!(axs, trialframetimestamps, 1:ntrials, pertrialmotion, c=:thermal, clims=clims)
    vline!(axs,[0,3],color=:white, lw=0.5)
    hline!(axs,[70],color=:white, lw=2)
    xlabel!(axs, "trial time [s]")
    ylabel!(axs, "visual       audio")
    title!(axs, "\n\nall trials n=$(ntrials)")

    axs = ax[2,2]
    heatmap!(axs, trialframetimestamps, 1:ntrials, pertrialmotionstationary, c=:thermal, clims=clims)
    vline!(axs,[0,3],color=:white, lw=0.5)
    hline!(axs,[70],color=:white, lw=2)
    xlabel!(axs, "trial time [s]")
    ylabel!(axs, "visual       audio")
    title!(axs,"\n\nstationary trials n=$(ntrials-nmotiontrialsdisplay)")
    # title!(axs,"\n\nproportion of timepoints alloved above threshold: $proportiontrialmotionthresholddisplay")

    axs = ax[2,3]
    heatmap!(axs, [0,1], [excludelevel; collect(0:0.05:2)], repeat([excludelevel; collect(0:0.05:2)],1,2), c=:thermal, clims=clims,
             xticks=nothing, aspectratio = 4)
    yticks!(axs, [excludelevel,0,0.5,1,1.5,2], ["excluded","0.0","0.5","1.0","1.5","2.0"])
    ylims!(axs, excludelevel, 2)
    xlims!(axs,-0.4,0.5)
    ylabel!(axs, "absolute motion [SU]\nmean over PCs")
    # plot!(axs, aspectratio = 0.25)
    

    go = [ events[1:ntrials÷2,:degree].==45; events[ntrials÷2+1:ntrials,:freq].==5000 ]
    success = .!events[!,:punish]

    for k in 1:3
        indices = (1:ntrials)
        # indices = (1:ntrials÷2, ntrials÷2+1:ntrials)[k]
        
        lim = (150,80,80)[k]
        N = (ntrials,ntrials÷2,ntrials÷2)[k]
        axs = ax[3,k]
        for (px,pr) in enumerate(proportiontrialmotionthresholds)
            if k==1
                for g in 1:2
                    mask = (go, .!go)[g]
                    golabel = ("go","nogo")[g]
                    gocolor = (:black,:red)[g]
                    N = sum(mask)
                    Ns = 1 .-sum(motionthresholdstrials[(px-1)*ntrials+1:px*ntrials,2:end][indices[mask],:], dims=1)' ./N
                    plot!(axs,  motionthresholds, Ns, label=["$golabel",nothing][Int(px>1)+1],  lw=2, color=gocolor,
                                alpha=1 .-(px-1.)/length(proportiontrialmotionthresholds), ylims=[0,1])
                end
            elseif k==2
                for c in 1:2
                    indices = (1:ntrials÷2, ntrials÷2+1:ntrials)[c]
                    contextlabel = ["visual context", "audio context"][c]
                    contextcolor = [:navy,:darkgreen][c]
                    N = length(indices)
                    Ns = 1 .-sum(motionthresholdstrials[(px-1)*ntrials+1:px*ntrials,2:end][indices,:], dims=1)' ./ N
                    # plot!(axs,  motionthresholds, Ns, label="prop=$pr", ylims=[0,lim],
                    #     xlabel="threshold [absolute motion]", ylabel="proportion of stationary trials")
                    plot!(axs,  motionthresholds, Ns, label=["$contextlabel",nothing][Int(px>1)+1],  lw=2, color=contextcolor,
                        alpha=1 .-(px-1.)/length(proportiontrialmotionthresholds), ylims=[0,1])
                end
            elseif k==3
                continue
                # for s in 1:2
                #     mask = (success, .!success)[s]
                #     successlabel = ("correct","error")[s]
                #     successcolor = (:green,:darkorange)[s]
                #     N = sum(mask)
                #     Ns = 1 .-sum(motionthresholdstrials[(px-1)*ntrials+1:px*ntrials,2:end][indices[mask],:], dims=1)' ./N
                #     plot!(axs,  motionthresholds, Ns, label=["$successlabel",nothing][Int(px>1)+1], lw=2, color=successcolor,
                #                 alpha=1 .-(px-1.)/length(proportiontrialmotionthresholds), ylims=[0,1])
                # end
            end
            yticks!(axs,[0,0.5,1])
            xlabel!(axs,"threshold [absolute motion]")
            ylabel!(axs,"fraction of stationary trials")
            hline!(axs,[1.],color=:grey, lw=0.5, ls=:dash, label=nothing)
            plot!(axs, legend=:bottomright)
        end
    end


    # show filters
    axs = ax[3,3]
    filternumbers = [1,2,3,5,12,13] # choose these filters
    Wm = getweights(P, S, chunkP, chunkS, timechunks)[filternumbers]     # use only the main pc Wms but not the residuals
    # Wm = [ Wm[1] Wm[2]; Wm[3] Wm[4]; Wm[5] Wm[6] ]
    b1 = 0.5.*ones(height,width÷6)
    b2 = 0.5.*ones(width÷6,2*width+3*width÷6)
    Wm = [ b2;
           b1 Wm[1] b1 Wm[2] b1;
           b2;
           b1 Wm[3] b1 Wm[4] b1;
           b2;
           b1 Wm[5] b1 Wm[6] b1;
           b2 ]
    
    Wm = RGB.(Wm)
    convertcolortoredgreen!(Wm,axes(Wm))

    plot!(axs,Wm, showaxis=false)


    panels="abcdefgh"
    ix = 0
    for hx in 1:3
        for wx in 1:3
            if hx==2 && wx==3 continue end
            ix += 1
            axs = ax[hx,wx]
            x1,x2 = xlims(axs)
            y1,y2 = ylims(axs)
            if hx+wx==6 y2,y1 = ylims(axs) end
            coords = [x2-x1,y2-y1] .* [-0.2,1.07] + [x1,y1]
            annotate!(axs, coords..., text(panels[ix], "Helvetica Bold", pointsize=26))
        end
    end

    display(ax)
    # savefig(ax, joinpath(outputpath,filename*"-pc-latent-motionenergy,threshold.png"))
    savefig(ax, joinpath("../../publish/journals/journal2020spring/figures/","Supp7-absolutemotion,threshold,proportionallowed,filters.png"))
    savefig(ax, joinpath("../../publish/journals/journal2020spring/figures/","Supp7-absolutemotion,threshold,proportionallowed,filters.pdf"))

    # datathresholds = [ ["proportion" motionthresholds']; motionthresholdstrials ]
    # @info "thresholds" datathresholds
    # writedlm(outputpath*filename*"-dpc-latent-motionenergy,threshold"*".csv",  datathresholds, ',')

    # write levels
    # @info "levels" motionlevelsmidpoints motionlevelstrials
    # writedlm(outputpath*filename*"-dpc-latent-motionenergy,levels"*".csv",  motionlevelsmidpoints, ',')
    # writedlm(outputpath*filename*"-dpc-latent-motionenergy,levelstrials"*".csv",  motionlevelstrials, ',')


    return z,movingtrialdisplay

end





