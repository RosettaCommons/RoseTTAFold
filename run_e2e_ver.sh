#!/bin/bash

# make the script stop when error (non-true exit code) is occured
set -e

############################################################
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup
# <<< conda initialize <<<
############################################################

#############
# help output
#############

function print_usage {
cat <<EOF

Usage: run_e2e_ver.sh -f <input fasta> [-w <working dir>][-c <cpus>][-m <mem>]

    -f, --fasta             Path to fasta input
    -w, --wd                Path to working directory (default: your current dir)
    -c, --cpus              CPU cores (default: 8)
    -m, --mem               Memory in Gigabytes (default: 64) 
    -d, --db                Database directory (default: pdb100_2021Mar03/pdb100_2021Mar03 in script dir)

EOF
exit 2
}

################
# default values
################

SCRIPT=`realpath -s $0`
export PIPEDIR=`dirname $SCRIPT`
WDIR=$(pwd)
CPU="8"  # number of CPUs to use
MEM="64" # max memory (in GB)
DB="$PIPEDIR/pdb100_2021Mar03/pdb100_2021Mar03"

##################
# argument parsing
##################

PARSED_ARGUMENTS=$(getopt -a -n run_e2e_ver.sh -o f:w:c:m:d: --long fasta:,wdir:,cpus:,mem:,db: -- "$@")
if [ $? != 0 ]; then
    print_usage    
fi

eval set -- "$PARSED_ARGUMENTS"
while :
do
    case "$1" in
        -f | --fasta)   IN="$2"     ; shift 2 ;;
        -w | --wdir)    WDIR="$2"   ; shift 2 ;;
        -c | --cpus)    CPU="$2"    ; shift 2 ;;
        -m | --mem)     MEM="$2"    ; shift 2 ;;
        -d | --db)      DB="$2"     ; shift 2;;
        -h | --help)    print_usage ;;
        # -- means the end of the args; drop this and break out of the while loop
        --) shift; break ;;
        *) print_usage ;;
    esac
done

if [ -z "$IN" ]; then
    echo -e "\n-f|--fasta is a required option"
    print_usage
fi

LEN=`tail -n1 $IN | wc -m`

mkdir -p $WDIR/log

conda activate RoseTTAFold
############################################################
# 1. generate MSAs
############################################################
if [ ! -s $WDIR/t000_.msa0.a3m ]
then
    echo "Running HHblits"
    $PIPEDIR/input_prep/make_msa.sh $IN $WDIR $CPU $MEM > $WDIR/log/make_msa.stdout 2> $WDIR/log/make_msa.stderr
fi


############################################################
# 2. predict secondary structure for HHsearch run
############################################################
if [ ! -s $WDIR/t000_.ss2 ]
then
    echo "Running PSIPRED"
    $PIPEDIR/input_prep/make_ss.sh $WDIR/t000_.msa0.a3m $WDIR/t000_.ss2 > $WDIR/log/make_ss.stdout 2> $WDIR/log/make_ss.stderr
fi


############################################################
# 3. search for templates
############################################################
if [ ! -s $WDIR/t000_.hhr ]
then
    echo "Running hhsearch"
    HH="hhsearch -b 50 -B 500 -z 50 -Z 500 -mact 0.05 -cpu $CPU -maxmem $MEM -aliw 100000 -e 100 -p 5.0 -d $DB"
    cat $WDIR/t000_.ss2 $WDIR/t000_.msa0.a3m > $WDIR/t000_.msa0.ss2.a3m
    $HH -i $WDIR/t000_.msa0.ss2.a3m -o $WDIR/t000_.hhr -atab $WDIR/t000_.atab -v 0 > $WDIR/log/hhsearch.stdout 2> $WDIR/log/hhsearch.stderr
fi


############################################################
# 4. end-to-end prediction
############################################################
if [ ! -s $WDIR/t000_.3track.npz ]
then
    echo "Running end-to-end prediction"
    python $PIPEDIR/network/predict_e2e.py \
        -m $PIPEDIR/weights \
        -i $WDIR/t000_.msa0.a3m \
        -o $WDIR/t000_.e2e \
        --hhr $WDIR/t000_.hhr \
        --atab $WDIR/t000_.atab \
        --db $DB 1> $WDIR/log/network.stdout 2> $WDIR/log/network.stderr
fi
echo "Done"
