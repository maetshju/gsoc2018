Running the training procedure for the model should be straightforward. Make sure that the WAV, Flux, CuArrays, JLD, and BSON packages are installed. As well, install [the fork I've made of the MFCC package](https://github.com/maetshju/MFCC.jl) (which only updates one line to make a function run on Julia 0.6). Start by cloning the Git repository for the project:

> $ git clone https://github.com/maetshju/gsoc2018.git

Navigate into the folder. To extract the data from the TIMIT corpus, use the `00-data.jl` script. More information on this script can be found in [the blog post dedicated to it](https://maetshju.github.io/speech-features.html).

> $ julia 00-data.jl

Now, to train the network, run the `02-speech-cnn.jl` script.

> $ julia 02-speech-cnn.jl

Note that it is essentially necessary to have a GPU to train the network on because the training process is extremely slow on just the CPU. Additionally, the script calls out to the GPU implementation of the CTC algorithm, which will fail without a GPU. The script will likely take over a day to run, so come back to it later. After the script finishes, the model should be trained and ready for use in making predictions.
