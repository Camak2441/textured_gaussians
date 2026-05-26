mkdir -p ../results/$1/$2
gcloud storage cp -r gs://diss-results/$1/$2/cfg.yml ../results/$1/$2
