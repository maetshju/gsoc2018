using Flux: onehotbatch
using WAV

# This is a custom fork of MFCC for the time being, since the project hasn't
# accepted my pull request that updated the depreacted `iceil` to `ceil`
# https://github.com/maetshju/MFCC.jl
using MFCC
using JLD

const TRAINING_DATA_DIR = "TIMIT/TRAIN"
const TEST_DATA_DIR = "TIMIT/TEST"

const TRAINING_OUT_DIR = "train"
const TEST_OUT_DIR = "test"

const PHONES = split("h#	q	eh	dx	iy	r	ey	ix	tcl	sh	ow	z	s	hh	aw	m	t	er	l	w	aa	hv	ae	dcl	y	axr	d	kcl	k	ux	ng	gcl	g	ao	epi	ih	p	ay	v	n	f	jh	ax	en	oy	dh	pcl	ah	bcl	el	zh	uw	pau	b	uh	th	ax-h	em	ch	nx	eng")
PHONE_TRANSLATIONS = Dict(phone=>i for (i, phone) in enumerate(PHONES))
PHONE_TRANSLATIONS["sil"] = PHONE_TRANSLATIONS["h#"]
const COLLAPSINGS = Dict(
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

function make_data(phn_fname, wav_fname)
    samps, sr = wavread(wav_fname)
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
    open(phn_fname, "r") do f
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
    labelInfoI = 1
    boundary, label = labelInfo[labelInfoI]
    nSegments = length(labelInfo)

    frameLengthSamples = FRAME_LENGTH * sr
    frameIntervalSamples = FRAME_INTERVAL * sr
    halfFrameLength = FRAME_LENGTH / 2

    seq = Vector()

    for i=1:size(fbanks)[1]
        win_end = frameLengthSamples + (i-1)*frameIntervalSamples

        if labelInfoI < nSegments && win_end - boundary > halfFrameLength

            labelInfoI += 1
            boundary, label = labelInfo[labelInfoI]
        end

        if label == "q"
            continue # delete windows labeled with q, as per paper
        end

        push!(seq, label)
    end

    fbank_deltas = deltas(fbanks)
    fbank_deltadeltas = deltas(fbank_deltas)
    features = hcat(fbanks, fbank_deltas, fbank_deltadeltas)
    return (features, seq)
end

function create_data(data_dir, out_dir)
    for (root, dirs, files) in walkdir(data_dir)
        phn_fnames = [x for x in files if contains(x, "PHN")]
        wav_fnames = [x for x in files if contains(x, "WAV")]

        one_dir_up = basename(root)
        println(root)

        for (phn_fname, wav_fname) in zip(phn_fnames, wav_fnames)
            phn_path = joinpath(root, phn_fname)
            wav_path = joinpath(root, wav_fname)

            x, y = make_data(phn_path, wav_path)

            y = [haskey(COLLAPSINGS, x) ? COLLAPSINGS[x]: x for x in y]
            y = [PHONE_TRANSLATIONS[x] for x in y]
            class_nums = [n for n in 1:61] # but only 39 used after folding
            y = onehotbatch(y, class_nums)

            base, _ = splitext(phn_fname)
            dat_name = one_dir_up * base * ".jld"
            dat_path = joinpath(out_dir, dat_name)
            save(dat_path, "x", x, "y", y)
        end
    end
end

create_data(TRAINING_DATA_DIR, TRAINING_OUT_DIR)
create_data(TEST_DATA_DIR, TEST_OUT_DIR)
