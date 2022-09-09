Contains code for figures and statistics for the publication:

Márton Albert Hajnal, Duy Tran, Michael Einstein, Mauricio Vallejo Martelo, Karen Safaryan, Pierre-Olivier Polack, Peyman Golshani, Gergő Orbán
"Continuous multiplexed population representations of task context in the mouse primary visual cortex"


This repo analyses the video capture, by reducing dimensionality, and also reconstructs the video from reduced pcas.


Written in julia. To install packages Project.toml and Manifest.toml are provided
in the root folder. Tested with julia 1.6 and 1.7.

Requires dvc (https://dvc.org) for data install, tested with versions 2.9-2.24.


src/ folder contents:
reducevideo.jl: main file, contains option parsing, and routing to routines
load.jl: loads raw video files, and encodes them in julia arrays
pca.jl: calculates the differential frames, and the dimension reducted video stream
annotation.jl: annotation routines for reconstruction
reconstruct.jl: reconstructs the video from the principal components


Install data into the local repo:
1. download the dvc cache zip file (movement-cache,dvc.zip) from the accompanying
zenodo link: https://zenodo.org/record/5045981/files/movement-cache,dvc.zip
to the repo root folder
2. unzip, it should unzip into .dvc/cache/* folders
3. populate the data tree MT020_2/ with data from the cache with the command at
the prompt: dvc checkout
4. check if everything is up to date, with the command: dvc status.


Run the analysis:
dvc.yaml contains the stages required to produce everything from pca to reconstruction, with the parameters params.yaml.
To test changes with params.yaml changes, do a full run with the command: dvc repro.
Individual stages can be run within julia REPL with:
args = [command,"MT020_2"]; include("src/reducevideo.jl")
where command can be inferred from dvc.yaml stages or in reducedvideo.jl, the if-elseif main command tree.


Questions about the code should be addressed to: hajnal.marton@wigner.mta.hu
