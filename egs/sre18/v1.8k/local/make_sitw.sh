#!/bin/bash
# Copyright 2017   David Snyder
#           2019   Johns Hopkins University (Jesus Villalba) 
# Apache 2.0

if [  $# != 3 ]; then
    echo "Usage: $0 <SITW_PATH> <fs> <OUTPATH>"
    echo "e.g. $0 /export/corpora/SRI/sitw 8 data/sitw"
    exit 1
fi
INPUTPATH=$1
fs=$2
OUTPUTPATH=$3

#TRAIN
for mode in dev eval; do
  OUTPATH=${OUTPUTPATH}_${mode}_enroll
  mkdir -p $OUTPATH 2>/dev/null
  WAVFILE=$OUTPATH/wav.scp
  SPKFILE=$OUTPATH/utt2spk
  MODFILE=$OUTPATH/utt2cond
  rm $WAVFILE $SPKFILE $MODFILE 2>/dev/null
  INPATH=${INPUTPATH}/$mode

  for enroll in core assist; do
    cat $INPATH/lists/enroll-${enroll}.lst | \
    sed 's/audio/audio-wav-'${fs}'KHz/g' | sed 's/flac/wav/g' |\
    while read line; do
      WAVID=`echo $line| awk '{print $2}' |\
        awk 'BEGIN{FS="[./]"}{print $(NF-1)}'`
      SPKID=`echo $line| awk '{print $1}'`
      WAV=`echo $line | awk '{print INPATH"/"$2}' INPATH=$INPATH`
      echo "${SPKID}_$WAVID $WAV" >> $WAVFILE
      echo "${SPKID}_$WAVID ${SPKID}" >> $SPKFILE
      echo "${SPKID}_$WAVID $enroll $mode" >> $MODFILE
    done
  done
  utils/fix_data_dir.sh $OUTPATH
done

for mode in dev eval; do
  #EVAL
  OUTPATH=${OUTPUTPATH}_${mode}_test
  mkdir -p $OUTPATH 2>/dev/null
  WAVFILE=$OUTPATH/wav.scp
  SPKFILE=$OUTPATH/utt2spk
  MODFILE=$OUTPATH/utt2cond
  rm $WAVFILE $SPKFILE $MODFILE 2>/dev/null
  mkdir -p $OUTPATH/trials 2>/dev/null
  mkdir -p $OUTPATH/trials/aux 2>/dev/null
  INPATH=${INPUTPATH}/$mode

  for trial in core multi; do
    cat $INPATH/lists/test-${trial}.lst | awk '{print $1,$2}' |\
      sed 's/audio/audio-wav-'${fs}'KHz/g'| sed 's/flac/wav/g' |\
    while read line; do
      WAVID=`echo $line | awk 'BEGIN{FS="[./]"} {print $(NF-1)}'`
      WAV=`echo $line | awk '{print INPATH"/"$1}' INPATH=$INPATH`
      echo "$WAVID $WAV" >> $WAVFILE
      echo "$WAVID $WAVID" >> $SPKFILE
      echo "$WAVID $trial $mode" >> $MODFILE
    done
  done

  for trial in core-core core-multi assist-core assist-multi; do
    cat $INPATH/keys/$trial.lst | sed 's@audio/@@g' | sed 's@.flac@@g' |\
    awk '{if ($3=="tgt")
           {print $1,$2,"target"}
         else
           {print $1,$2,"nontarget"}
         }'   > $OUTPATH/trials/${trial}.lst
  done

  for trial in $INPATH/keys/aux/* ; do
    trial_name=`basename $trial`
    cat $trial | sed 's@audio/@@g' | sed 's@.flac@@g' |\
    awk '{if ($3=="tgt")
           {print $1,$2,"target"}
         else
           {print $1,$2,"nontarget"}
     }'   > $OUTPATH/trials/aux/${trial_name}
  done
  utils/fix_data_dir.sh $OUTPATH
done
