#!/bin/bash

#SBATCH --account=rubin:developers      # Account name
#SBATCH --partition=milano
#
#SBATCH --job-name=ddf_sd53
#SBATCH --output=output-%j.txt
#SBATCH --error=output-%j.txt
#
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=40g
#
#SBATCH --time=3-14:10:00
#SBATCH --chdir=/sdf/data/rubin/shared/fbs_sims/sims_featureScheduler_runs5.3/ddf_sd
#SBATCH --mail-user=yoachim@uw.edu
#
##turn on e-mail notification
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yoachim@uw.edu

## Set up the evironment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rubin13
export OPENBLAS_NUM_THREADS=1

python ddf_sd.py


# rm maf.sh
#  
# ls *_10yrs.db | xargs -I'{}' echo "scimaf_dir --db '{}'" > maf.sh
# ls *_10yrs.db | xargs -I'{}' echo "glance_dir --db '{}'" >> maf.sh
# ls *_10yrs.db | xargs -I'{}' echo "ddf_dir --db '{}'" >> maf.sh
# ls *_10yrs.db | xargs -I'{}' echo "metadata_dir --db '{}'" >> maf.sh
# 
# generate_ss
# cat ss_script.sh >> maf.sh
#  
# cat maf.sh | parallel -j 5
# 