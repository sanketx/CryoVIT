#!/bin/bash

# Check if four arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 model (cryovit, unet3d, sam2, or medsam) label_key (microtubule, cristae, granule, bacteria)"
    exit 1
fi

case $2 in
    "mito")
        exp_name="fractional_mito"
        ;;
    "mito_membrane")
        exp_name="fractional_mito_membrane"
        ;;
    "microtubule")
        exp_name="fractional_microtubule"
        ;;
    "cristae")
        exp_name="fractional_cristae"
        ;;
    "granule")
        exp_name="fractional_granule"
        ;;
    "bacteria")
        exp_name="fractional_bacteria"
        ;;
    *)
        echo "Invalid label_key. Choose from: microtubule, cristae, granule, bacteria."
        exit 1
        ;;
esac

max_jobs=1024  # Maximum concurrent jobs
total_jobs=$(( ${#samples[@]} * 10 ))
current_job=0

for split_id in {0..10}; do
    for fraction_id in {10..10}; do  # hardcoded to 100% data
        # Check the number of running jobs
        while [ $(squeue -u $USER --noheader | wc -l) -ge $max_jobs ]; do
            sleep 10  # Wait for 10 seconds before checking again
        done

        exp_cmd="$(dirname "$0")/fractional_experiment_job.sh $exp_name $split_id $fraction_id $1 $2"
        job_name="fractional_experiment_${1}_${2}_${split_id}_${fraction_id}"
        out_dir="$(dirname "$0")/outputs"

        sbatch \
            --partition="ampere" \
            --account="cryoem:C036" \
            --job-name="$job_name" \
            --output="${out_dir}/${job_name}.out" \
            --ntasks=1 \
            --cpus-per-task=8 \
            --mem-per-cpu=12gb \
            --gres=gpu:a100:1 \
            --time=3:00:00 \
            --wrap="$exp_cmd"

        ((current_job++))
        echo Job $current_job / $total_jobs: \
            fraction=$fraction_id, \
            split_id=$split_id
    done
done