#!/bin/bash

# Check if four arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 sample group (ad, hd, rgc, algae, campy) model (cryovit, cryovit_highfreq, unet3d, sam2, or medsam) label_key (mito, microtubule, cristae, granule, dense_mito)"
    exit 1
fi

# Check for valid labels for the specified group
if [ "$1" != "hd" ] && [ "$3" == "granule" -o "$3" == "dense_mito" ]; then
    echo "Label keys granule and dense_mito are only valid for the hd group."
    exit 1
fi

case $1 in
    "ad")
        samples=(
            "AD"
            "AD_Abeta"
            "Aged"
            "Young"
        )
        ;;
    "hd")
        case $3 in
            "mito")
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
                )
                ;;
            "dense_mito")
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
                )
                ;;
            "microtubule")
                samples=(
                    "BACHD_Microtubules"
                    "Q109_Microtubules"
                    "Q18_Microtubules"
                    "WT_Microtubules"
                )
                ;;
            "cristae")
                samples=(
                    "Q18"
                    "Q53"
                )
                ;;
            "granule")
                samples=(
                    "BACHD"
                    "dN17_BACHD"
                    "Q109"
                    "Q18"
                    "Q20"
                    "Q53"
                    "Q66"
                    "Q66_GRFS1"
                    "WT"
                )
                ;;
            *)
                echo "Invalid label_key for hd group. Choose from: mito, microtubule, cristae."
                exit 1
                ;;
        esac
        ;;
    "rgc")
        samples=(
            "RGC_CM"
            "RGC_control"
            "RGC_naPP"
            "RGC_PP"
        )
        ;;
    "algae")
        samples=(
            "CZI_Algae"
        )
        ;;
    "campy")
        samples=(
            "CZI_Campy_C"
            "CZI_Campy_CDel"
            "CZI_Campy_F"
        )
        ;;
    *)
        echo "Invalid group. Choose from: ad, hd, campy, bacteria."
        exit 1
        ;;
esac

max_jobs=1024  # Maximum concurrent jobs
total_jobs=$(( ${#samples[@]} * 10 ))
current_job=0

for sample in "${samples[@]}"; do
    for split_id in {0..9}; do
        # Check the number of running jobs
        while [ $(squeue -u $USER --noheader | wc -l) -ge $max_jobs ]; do
            sleep 10  # Wait for 10 seconds before checking again
        done

        exp_cmd="$(dirname "$0")/single_sample_job.sh $sample $split_id $1 $2 $3"
        job_name="single_sample_${2}_${3}_${sample}_${split_id}"
        out_dir="$(dirname "$0")/outputs"

        sbatch \
            --partition="ampere" \
            --account="cryoem:C073" \
            --job-name="$job_name" \
            --output="${out_dir}/${job_name}.out" \
            --ntasks=1 \
            --cpus-per-task=8 \
            --mem-per-cpu=8gb \
            --gres=gpu:a100:1 \
            --time=01:00:00 \
            --wrap="$exp_cmd"

        ((current_job++))
        echo Job $current_job / $total_jobs: \
            sample=$sample, \
            split_id=$split_id
    done
done