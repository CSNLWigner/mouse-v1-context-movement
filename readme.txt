Contains code for figures and statistics for the publication:

Márton Albert Hajnal, Duy Tran, Michael Einstein, Mauricio Vallejo Martelo, Karen Safaryan,
Pierre-Olivier Polack, Peyman Golshani, Gergő Orbán
"Continuous multiplexed population representations
of task context in the mouse primary visual cortex"


This repo analyses the video capture, by reducing dimensionality, or from body parts, and also reconstructs the video from reduced pcas.


Written in julia. To install packages Project.toml and Manifest.toml are provided
in the root folder. Tested with julia 1.8.

Requires dvc (https://dvc.org) for data install, tested with versions 2.50.


src/ folder content:
reducevideo.jl: main file, contains option parsing, and routing to routines
load.jl: loads raw video files, and encodes them in julia arrays
pca.jl: calculates the differential frames, the dimension reducted video stream from
        both the abstolute frames (posture) and the differential frames (motion),
        threshold command provides Supp. Fig. 7.
bodyparts.jl: analyse motion levels of body parts in regions of interest in the differential
        bodyparts command provides Supp. Fig. 8.
annotation.jl: annotation routines for reconstruction
reconstruct.jl: reconstructs the video from the differential principal components


Install data into the local repo:
1. download into the root folder the dvc cache zip file movement-cache,dvc.zip
from the accompanying zenodo repository (all 6 parts):
doi://10.5281/zenodo.7900224
or use wget with files:
wget https://zenodo.org/record/7900224/files/movement-cache,dvc.{z01,z02,z03,z04,z05,zip}
2. unzip, it should unzip into .dvc/cache/* folders
3. populate the data tree MT020_2/ with data from the cache with the command at
the prompt: dvc checkout
4. check if everything is up to date, with the command: dvc status


Run the analysis:
dvc.yaml contains the stages required to produce everything from pca to reconstruction,
with the parameters params.yaml.
To test changes within params.yaml, do a full run with the command: dvc repro,
or individual stages, e.g.: dvc repro pca.
Individual stages can be run within julia REPL as well, with:
args = [command,"MT020_2"]; include("src/reducevideo.jl")
where command can be inferred from dvc.yaml stages (such as pca)
or from the if-elseif main command exacution tree in reducevideo.jl.
All output csv and png files are needed for the sibling repository:
mouse-v1-context for the python scripts of the main figures.


Questions about the code should be addressed to: hajnal.marton@wigner.mta.hu
