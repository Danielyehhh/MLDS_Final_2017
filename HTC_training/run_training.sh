wget -O "training_model/FCN_modelv3vv.h5" "https://www.dropbox.com/s/56dnyiub5h1c2lc/FCN_modelv3vv.h5?dl=0"
python FCNTRAINING-1.py $1 $2
