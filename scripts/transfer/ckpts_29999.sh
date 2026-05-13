mkdir -p ../results/$1/$2/ckpts
gcloud storage cp gs://diss-results/$1/$2/ckpts/cktp_29999.pt ../results/$1/$2/clpts
