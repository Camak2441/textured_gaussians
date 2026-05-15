mkdir -p ../results/$1/$2/ckpts
gcloud storage cp gs://diss-results/$1/$2/ckpts/ckpt_29999.pt ../results/$1/$2/ckpts
