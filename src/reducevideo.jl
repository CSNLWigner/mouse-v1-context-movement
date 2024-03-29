__precompile__()
module reducevideo

using YAML


using VideoIO
using VideoIO: N0f8
using ImageCore: channelview, colorview
using Images: save

using Dates
using DataFrames
using Serialization
using CSV
using DelimitedFiles: writedlm

using DSP
using MultivariateStats
using Distributions
using StatsBase: mean, std, crosscor, Histogram, fit, kldivergence
using Random: shuffle, shuffle!
using LinearAlgebra
using Clustering: kmeans!, assignments
using Interpolations
using Optim

using Plots
# import NeuroscienceCommon.Figs.__init__
# using NeuroscienceCommon.MathUtils: convolve, erode1d!, dilate1d!



isinteractive() ? args = Main.args : args = ARGS
command = args[1]
mouseid = args[2]
if length(args)>=3 videoid = args[3] else videoid = mouseid end

config = YAML.load_file("params.yaml", dicttype=Dict{Symbol,Any})
@info "config YAML" config


if command=="cachecodec"
    video = config[:videos][Symbol(videoid)]
    filename = videoid
    crop = video[:crop]
    framestart = video[:framestart]           #  # 4776         multimodal start
    frameduration = video[:frameduration]   #6000 # 9450                      #14226-framestart # 14221 (last multimodal trial in video)
elseif command in ["pca", "dpca", "display", "threshold", "reconstruct", "bodyparts"]
    session = config[:sessions][Symbol(mouseid)]
    videos = config[:videos]
    filename = mouseid
    crop = videos[Symbol(session[:videos][1])][:crop]
end








rawvideorootpath = config[:rawvideorootpath]
rawbehaviourrootpath = config[:rawbehaviourrootpath]
ext = config[:videofileextension]
inputpath = joinpath(rawvideorootpath, mouseid*"/")
outputpath = joinpath(rawvideorootpath, mouseid*"/pca/")
@info "config" filename inputpath outputpath

# crop = parse.(Int64,args[3:3+4-1])            # [960, 1536, 200, 600]
chunksize = config[:chunksize]
width,height = (crop[2]-crop[1]), (crop[4]-crop[3])
@info "framesize" height width
fps = 0
# inputdims = width * height
chunkdim = config[:chunkdim]
maxpc = config[:maxpc]
reconstructpc = config[:reconstructpc]
downsample = config[:downsample]
exportframestart = config[:exportframestart]
exportframeduration = config[:exportframeduration]
# global timestamps
timestamps = Float32[]
# background difference smoothing
backgroundrate = Float32(config[:backgroundrate])
diffimagerate = Float32(config[:diffimagerate])
# movement stationary threshold
kernelhalfwidth = config[:kernelhalfwidth]
motionthresholds = config[:motionthresholds]
proportiontrialmotionthresholds = config[:proportiontrialmotionthresholds]
exportframeorientation = config[:exportframeorientation]
exportoverlayframe = config[:exportoverlayframe]

bodypartnames = config[:bodypartnames]
bodypartboundaries = config[:bodypartboundaries][Symbol(mouseid)]

# first-order trial start-end impulse response
# const impulseresponsehalfwidth = 5
# const impulseresponsewidth = 2*impulseresponsehalfwidth+1
# signal = collect([1:1.:impulseresponsehalfwidth; (impulseresponsehalfwidth+1):-1.:1])
# const impulseresponse = signal/sum(signal)

# zero-order trial start-end impulse response
const impulseresponsewidth = 15*3
const impulseresponse = ones(impulseresponsewidth)




include("load.jl")
include("utils.jl")
include("annotation.jl")
include("pca.jl")
include("reconstruct.jl")
include("bodyparts.jl")








function main()

    


    if command=="cachecodec"
        fps,timestamps,X = loadvideo(joinpath(inputpath, filename*ext), crop)
        X = Gray.(reinterpret(N0f8, X))
        serialize(joinpath(outputpath,filename*"-cropped-raw"*".dat"),(fps,timestamps,X))




    elseif command=="pca"        # absolute intensity


        fps,timestamps,X = concatenateblocks()

        if true         # quick cacheing (use false here 2nd time) outside dvc
            Z,P,S,μ,σ,chunkZ,chunkP,chunkS,chunkμ,chunkσ,timechunks = pcatransform(X)
            serialize(joinpath(outputpath,filename*"-pc-latent"*".dat"),(fps,timestamps,Z,P,S,μ,σ,chunkZ,chunkP,chunkS,chunkμ,chunkσ,timechunks))
            
        else
            (fps,timestamps,Z,P,S,μ,σ,chunkZ,chunkP,chunkS,chunkμ,chunkσ,timechunks) = deserialize(joinpath(outputpath,filename*"-pc-latent"*".dat"))
        end

        exportpctimeseries(Z,timestamps,"pc")




               
    elseif command=="dpca"          # differential intensity

        fps,timestamps,X = concatenateblocks()

        if true         # quick cacheing (use false here 2nd time) outside dvc
            smoothbaselinedifference!(X)
            dZ,dP,dS,dμ,dσ,dchunkZ,dchunkP,dchunkS,dchunkμ,dchunkσ,dtimechunks = pcatransform(X)
            serialize(joinpath(outputpath,filename*"-dpc-latent"*".dat"),(fps,timestamps,dZ,dP,dS,dμ,dσ,dchunkZ,dchunkP,dchunkS,dchunkμ,dchunkσ,dtimechunks))
        else
            (fps,timestamps,dZ,dP,dS,dμ,dσ,dchunkZ,dchunkP,dchunkS,dchunkμ,dchunkσ,dtimechunks) = deserialize(joinpath(outputpath,filename*"-dpc-latent"*".dat"))
        end

        exportpctimeseries(dZ,timestamps,"dpc")




    elseif command=="display"

        fps,timestamps,Z,P,S,μ,σ,chunkZ,chunkP,chunkS,chunkμ,chunkσ,timechunks = deserialize(joinpath(outputpath,filename*"-pc-latent"*".dat"))
        fps,timestamps,dZ,dP,dS,dμ,dσ,dchunkZ,dchunkP,dchunkS,dchunkμ,dchunkσ,dtimechunks = deserialize(joinpath(outputpath,filename*"-dpc-latent"*".dat"))

        displaypctimeseries(Z,timestamps,"pc")
        displaypctimeseries(dZ,timestamps,"dpc")




    elseif command=="threshold"

        fps,timestamps,Z,P,S,μ,σ,chunkZ,chunkP,chunkS,chunkμ,chunkσ,timechunks = deserialize(joinpath(outputpath,filename*"-dpc-latent"*".dat"))

        fa = annotateframes(timestamps)
        # diagnosetrialboundaries(timestamps, fa); return

        z,movingtrial = threshold(fps,timestamps,Z,P,S,μ,σ,chunkZ,chunkP,chunkS,chunkμ,chunkσ,timechunks)

        # fps,timestamps,X = concatenateblocks()
        # fa = annotateframes(timestamps)

        # viewstationarytrials(X, movingtrial, timestamps, fps, fa)



    elseif command=="reconstruct"

        fps,timestamps,Z,P,S,μ,σ,chunkZ,chunkP,chunkS,chunkμ,chunkσ,timechunks = deserialize(joinpath(outputpath,filename*"-dpc-latent"*".dat"))
        fps,timestamps,X = concatenateblocks()
        fa = annotateframes(timestamps)

        reconstruct(X,Z,P,S,μ,σ,chunkZ,chunkP,chunkS,chunkμ,chunkσ,timechunks,fps,fa)


    elseif command=="bodyparts"
        
        if true           # on run non-dvc-cache
            fps,timestamps,X = concatenateblocks()
            exampleframe = copy(X[:,14333])
            smoothbaselinedifference!(X)
            serialize(joinpath(outputpath,filename*"-diff-frames"*".dat"),(fps,timestamps,X,exampleframe))
        else
            (fps,timestamps,X,exampleframe) = deserialize(joinpath(outputpath,filename*"-diff-frames"*".dat"))
        end
        
        movementrepertoire(X,exampleframe,timestamps)
    
    end

end





main()


end
