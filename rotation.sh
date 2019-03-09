#!/bin/bash

set -x

RUN_FILE=rnn-cv.py

LOG_DIR="logs"
OUT_FILE="data-v9.txt"

RANDOM=$$$(date +%s)
# Parameters
num_neurons_v=( 35 100 140 ) # "70" "100" "140" ) # "35" 
num_neurons2_v=( 5 14 ) # "5" "7" "14" "35" ) # "10" 
rec_dropout_v=( "0.0" "0.01" "0.1" "0.2" "0.5" )
dropout_v=( "0.0" "0.1" "0.5" )
lr_v=( "0.01" "0.005" ) # "0.005" "0.01" "0.05" )
n_splits_v=( 10 14 ) # 10 14 35
patience_v=( 10 ) #
batch_size_v=( 20 ) # 10 20 )


# Preamble
mkdir $LOG_DIR
echo "Starting rotation with parameters:" >> "$OUT_FILE"
echo "num_neurons:   " "${num_neurons_v[@]}"   >> "$OUT_FILE"
echo "num_neurons2:  " "${num_neurons2_v[@]}"  >> "$OUT_FILE"
echo "rec_dropout:   " "${rec_dropout_v[@]}"   >> "$OUT_FILE"
echo "dropout:       " "${dropout_v[@]}"       >> "$OUT_FILE"
echo "lr:            " "${lr_v[@]}"            >> "$OUT_FILE"
echo "n_splits:      " "${n_splits_v[@]}"      >> "$OUT_FILE"
echo "patience:      " "${patience_v[@]}"      >> "$OUT_FILE"
echo "batch_size:    " "${batch_size_v[@]}"    >> "$OUT_FILE"

# Launch
echo "START";


for num_neurons in "${num_neurons_v[@]}"; do
    for num_neurons2 in "${num_neurons2_v[@]}"; do
        for rec_dropout in "${rec_dropout_v[@]}"; do
            for dropout in "${dropout_v[@]}"; do
                for lr in "${lr_v[@]}"; do
                    for n_splits in "${n_splits_v[@]}"; do
                        for patience in "${patience_v[@]}"; do
                            for batch_size in "${batch_size_v[@]}"; do


#for gangen in $(seq 1 100); do
#num_neurons=${num_neurons_v[$RANDOM % ${#num_neurons_v[@]} ]}
#num_neurons2=${num_neurons2_v[$RANDOM % ${#num_neurons2_v[@]} ]}
#rec_dropout=${rec_dropout_v[$RANDOM % ${#rec_dropout_v[@]} ]}
#dropout=${dropout_v[$RANDOM % ${#dropout_v[@]} ]}
#lr=${lr_v[$RANDOM % ${#lr_v[@]} ]}
#n_splits=${n_splits_v[$RANDOM % ${#n_splits_v[@]} ]}
#patience=${patience_v[$RANDOM % ${#patience_v[@]} ]}
#batch_size=${batch_size_v[$RANDOM % ${#batch_size_v[@]} ]}



EX_ID="$num_neurons\t$num_neurons2\t$rec_dropout\t$dropout\t$lr\t$n_splits\t$patience\t$batch_size"
FILE_ID="$LOG_DIR/$num_neurons-$num_neurons2-$rec_dropout-$dropout-$lr-$n_splits-$patience-$batch_size.log"

echo "$EX_ID";

#if [ ! -f "$FILE_ID" ]; then
python "$RUN_FILE" "shifted" "$num_neurons" "$num_neurons2" "$rec_dropout" "$dropout" "$lr" "$n_splits" "$patience" "$batch_size" > "$FILE_ID"
all_good=$?
#else
#    all_good=0
#fi

accuracy=`sed -rn "s/^.*Accuracy: ([0-9\.]*)\% \(.*- ([0-9\.]*)\%\).*$/\1\t\2/p" "$FILE_ID"`

out_string="$all_good\t$accuracy\t|\t$EX_ID"
printf "$out_string\n" >> "$OUT_FILE"



#done

                            done
                        done
                    done
                done
            done
        done
    done
done

vim "$OUT_FILE"
exit 0