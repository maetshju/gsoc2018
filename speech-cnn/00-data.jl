using Flux: onehotbatch
using WAV
using JLD

# This is a custom fork of MFCC for the time being, since the project hasn't
# accepted my pull request that updated the depreacted `iceil` to `ceil`
# https://github.com/maetshju/MFCC.jl
using MFCC

# Define constants that will be used
const TRAINING_DATA_DIR = "TIMIT/TRAIN"
const TEST_DATA_DIR = "TIMIT/TEST"

const TRAINING_OUT_DIR = "train"
const TEST_OUT_DIR = "test"

# Make dictionary to map from phones to class numbers
const PHONES = split("h#	q	eh	dx	iy	r	ey	ix	tcl	sh	ow	z	s	hh	aw	m	t	er	l	w	aa	hv	ae	dcl	y	axr	d	kcl	k	ux	ng	gcl	g	ao	epi	ih	p	ay	v	n	f	jh	ax	en	oy	dh	pcl	ah	bcl	el	zh	uw	pau	b	uh	th	ax-h	em	ch	nx	eng")
translations = Dict(phone=>i for (i, phone) in enumerate(PHONES))
translations["sil"] = translations["h#"]
const PHONE_TRANSLATIONS = translations

# Make dictionary to perform class folding
const FOLDINGS = Dict(
    "ao" => "aa",
    "ax" => "ah",
    "ax-h" => "ah",
    "axr" => "er",
    "hv" => "hh",
    "ix" => "ih",
    "el" => "l",
    "em" => "m",
    "en" => "n",
    "nx" => "n",
    "eng" => "ng",
    "zh" => "sh",
    "pcl" => "sil",
    "tcl" => "sil",
    "kcl" => "sil",
    "bcl" => "sil",
    "dcl" => "sil",
    "gcl" => "sil",
    "h#" => "sil",
    "pau" => "sil",
    "epi" => "sil",
    "ux" => "uw"
)

# The frame length and interval are guesses because the paper does not
# specify them. Most literature seems to use these values, however.
FRAME_LENGTH = 0.025 # ms
FRAME_INTERVAL = 0.010 # ms

"""
    makeFeatures(wavFname, phnFname)

Extracts Mel filterbanks and associated labels from `wavFname` and `phnFaname`.
"""
function makeFeatures(phnFname, wavFname)
    samps, sr = wavread(wavFname)
    samps = vec(samps)

    frames = powspec(samps, sr; wintime=FRAME_LENGTH, steptime=FRAME_INTERVAL)

    # I am assuming we want log energy because Mel scale is a log scale
    # If not, we can remove the log

    # Transpose so that we get each row as an observation of variables, as
    # opposed to each column as an observation of variables; not that one is
    # inherently better than the other, but most work I've encountered seems to
    # prefer rows as observations, and that's how I think about the problems too
    energies = log.(sum(frames', 2))

    fbanks = audspec(frames, sr; nfilts=40, fbtype=:mel)'
    fbanks = hcat(fbanks, energies)

    local lines
    open(phnFname, "r") do f
        lines = readlines(f)
    end

    boundaries = Vector()
    labels = Vector()

    # first field in the file is the beginning sample number, which isn't
    # needed for calculating where the labels are
    for line in lines
        _, boundary, label = split(line)
        boundary = parse(Int64, boundary)
        push!(boundaries, boundary)
        push!(labels, label)
    end

    labelInfo = collect(zip(boundaries, labels))
    labelInfoIdx = 1
    boundary, label = labelInfo[labelInfoIdx]
    nSegments = length(labelInfo)

    frameLengthSamples = FRAME_LENGTH * sr
    frameIntervalSamples = FRAME_INTERVAL * sr
    halfFrameLength = FRAME_LENGTH / 2

    # Begin generating sequence labels by looping through the acoustic
    # sample numbers

    labelSequence = Vector() # Holds the sequence of labels

    idxsToDelete = Vector() # To store indices for frames labeled as 'q'
    for i=1:size(fbanks, 1)
        win_end = frameLengthSamples + (i-1)*frameIntervalSamples

        # Move on to next label if current frame of samples is more than half
        # way into next labeled section and there are still more labels to
        # iterate through
        if labelInfoIdx < nSegments && win_end - boundary > halfFrameLength

            labelInfoIdx += 1
            boundary, label = labelInfo[labelInfoIdx]
        end

        if label == "q"
            push!(idxsToDelete, i)
            continue
        end

        push!(labelSequence, label)
    end

    # Remove the frames that were labeld as 'q'
    fbanks = fbanks[[i for i in 1:size(fbanks,1) if !(i in Set(idxsToDelete
    ))],:]

    fbank_deltas = deltas(fbanks)
    fbank_deltadeltas = deltas(fbank_deltas)
    features = hcat(fbanks, fbank_deltas, fbank_deltadeltas)
    return (features, labelSequence)
end

"""
    createData(data_dir, out_dir)

Extracts data from files in `data_dir` and saves results in `out_dir`.
"""
function createData(data_dir, out_dir)
    for (root, dirs, files) in walkdir(data_dir)

        # Exclude the files that are part of the speaker accent readings
        files = [x for x in files if ! contains(x, "SA")]

        phnFnames = [x for x in files if contains(x, "PHN")]
        wavFnames = [x for x in files if contains(x, "WAV")]

        one_dir_up = basename(root)
        println(root)

        for (wavFname, phnFname) in zip(wavFnames, phnFnames)
            phn_path = joinpath(root, phnFname)
            wav_path = joinpath(root, wavFname)

            x, y = makeFeatures(phn_path, wav_path)

            # Perform label foldings
            y = [haskey(FOLDINGS, x) ? FOLDINGS[x]: x for x in y]
            y = [PHONE_TRANSLATIONS[x] for x in y]

            # Generate class nums; there are 61 total classes, but only 39 are
            # used after folding. However, because we're using connectionist
            # temporal classification loss, we need an extra class, so we go
            # up to 62.
            class_nums = [n for n in 1:62]
            y = onehotbatch(y, class_nums)'

            base, _ = splitext(phnFname)
            dat_name = one_dir_up * base * ".jld"
            dat_path = joinpath(out_dir, dat_name)
            save(dat_path, "x", x, "y", y)
        end
    end
end

createData(TRAINING_DATA_DIR, TRAINING_OUT_DIR)
createData(TEST_DATA_DIR, TEST_OUT_DIR)
