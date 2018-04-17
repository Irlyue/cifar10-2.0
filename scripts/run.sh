function train_n_epochs
{
    python train.py
        --lr=1e-3 \
        --batch_size=$BATCH_SIZE \
        --n_epochs=$1 \
        --model_dir=$MODEL_DIR
}

function pre_run
{
    cd ..
    if [ -e $MODEL_DIR ]; then
        rm -rf $MODEL_DIR
    fi
}

MODEL_DIR=/tmp/cifar10
BATCH_SIZE=64
TRAIN_EPOCHS=10


if [ "$1" == "train" ]; then
    pre_run
    train_n_epochs $TRAIN_EPOCHS
elif [ "$1" == "eval" ]; then
    eval_forever
elif ["$1" == "eval_once"]; then
    echo "eval_once"
else
    echo "$1" not allowed
fi
