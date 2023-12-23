#!/bin/bash

# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 model (cryovit or unet3d) label_key (mito or mito_ai)"
    exit 1
fi

samples=(
    "BACHD"
    "dN17_BACHD"
    "Q109"
    "Q18"
    "Q20"
    "Q53"
    "Q53_KD"
    "Q66"
    "Q66_GRFS1"
    "Q66_KD"
    "WT"
    "cancer"
)

max_jobs=24  # Maximum concurrent jobs
total_jobs=$(( ${#samples[@]} * 10 ))
current_job=0

for sample in "${samples[@]}"; do
    for split_id in {1..10}; do
        # Check the number of running jobs
        while [ $(squeue -u $USER --noheader | wc -l) -ge $max_jobs ]; do
            sleep 10  # Wait for 10 seconds before checking again
        done

        exp_cmd="$(dirname "$0")/fractional_sample_job.sh $sample $split_id $1 $2"
        job_name="single_sample_${sample}_${split_id}_${1}_${2}"
        out_dir="$(dirname "$0")/outputs"

        sbatch \
            --partition="cryoem" \
            --job-name="$job_name" \
            --output="${out_dir}/${job_name}.out" \
            --ntasks=1 \
            --cpus-per-task=8 \
            --mem-per-cpu=6gb \
            --gres=gpu:a100:1 \
            --time=02:00:00 \
            --wrap="$exp_cmd"

        ((current_job++))
        echo Job $current_job / $total_jobs: \
            sample=$sample, \
            split_id=$split_id
    done
done