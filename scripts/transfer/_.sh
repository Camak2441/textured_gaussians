mkdir -p ../results/$1/$2
gcloud storage cp -r gs://diss-results/$1/$2/cfg.yml ../results/$1/$2
gcloud storage cp -r gs://diss-results/$1/$2/stats ../results/$1/$2
gcloud storage cp -r gs://diss-results/$1/$2/renders ../results/$1/$2
gcloud storage cp -r gs://diss-results/$1/$2/videos ../results/$1/$2
