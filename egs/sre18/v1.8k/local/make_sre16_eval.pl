#!/usr/bin/perl
use warnings; #sed replacement for -w perl parameter
# Copyright 2017   David Snyder
#           2019   Johns Hopkins University (Jesus Villalba) 
# Apache 2.0
#

if (@ARGV != 3) {
  print STDERR "Usage: $0 <path-to-SRE16-eval> <fs> <path-to-output>\n";
  print STDERR "e.g. $0 /export/corpora/SRE/R149_0_1 8 data/\n";
  exit(1);
}

($db_base, $fs, $out_dir) = @ARGV;


# Make call2utt dictionary
open(U2C, "<$db_base/docs/sre16_eval_enrollment_segment_key.tsv") || die "Could not open file $db_base/docs/sre16_eval_enrollment_segment_key.tsv.  It might be located somewhere else in your distribution.";
%utt2call = ();
while(<U2C>) {
  chomp;
  $line = $_;
  @toks = split(" ", $line);
  $call = $toks[1];
  $utt = $toks[0];
  if ($utt ne "segment") {
      $utt2call{$utt} = $call;
  }

}
close(U2C) || die;

open(SEG_KEY, "<$db_base/docs/sre16_eval_segment_key.tsv") || die "Could not open trials file $db_base/docs/sre16_eval_segment_key.tsv.  It might be located somewhere else in your distribution.";

while(<SEG_KEY>) {
  chomp;
  $line = $_;
  @toks = split(" ", $line);
  $utt = $toks[0];
  $call = $toks[1];
  if ($utt ne "segment") {
    $utt2call{$utt} = $call;
  }
}
close(SEG_KEY) || die;



open(SPK_KEY, "<$db_base/metadata/call_sides.tsv") || die "Could not open file $db_base/metadata/call_sides.tsv.  It might be located somewhere else in your distribution.";
%call2spk = ();
while(<SPK_KEY>) {
  chomp;
  $line = $_;
  @toks = split(" ", $line);
  $call = $toks[0];
  $spk = $toks[2];
  if ($call ne "call_id"){ 
      $call2spk{$call} = $spk;
  }

}
close(SPK_KEY) || die;


open(LANG_KEY, "<$db_base/metadata/calls.tsv") || die "Could not open file $db_base/metadata/calls.tsv.  It might be located somewhere else in your distribution.";
%call2lang = ();
while(<LANG_KEY>) {
  chomp;
  $line = $_;
  @toks = split(" ", $line);
  $call = $toks[0];
  $lang = $toks[1];
  if ($call ne "call_id"){
      $call2lang{$call} = $lang;
  }	  
}
close(LANG_KEY) || die;


# Handle enroll
$out_dir_enroll = "$out_dir/sre16_eval_enroll";
if (system("mkdir -p $out_dir_enroll")) {
  die "Error making directory $out_dir_enroll";
}

$tmp_dir_enroll = "$out_dir_enroll/tmp";
if (system("mkdir -p $tmp_dir_enroll") != 0) {
  die "Error making directory $tmp_dir_enroll";
}


open(SPKR, ">$out_dir_enroll/utt2spk") || die "Could not open the output file $out_dir_enroll/utt2spk";
open(MODEL, ">$out_dir_enroll/utt2model") || die "Could not open the output file $out_dir_enroll/utt2model";
open(U2L, ">$out_dir_enroll/utt2lang") || die "Could not open the output file $out_dir_enroll/utt2lang";
open(WAV, ">$out_dir_enroll/wav.scp") || die "Could not open the output file $out_dir_enroll/wav.scp";
open(META, "<$db_base/docs/sre16_eval_enrollment.tsv") or die "cannot open wav list";
%utt2fixedutt = ();
while (<META>) {
  $line = $_;
  @toks = split(" ", $line);
  $model = $toks[0];
  $utt = $toks[1];
  if ($utt ne "segment") {
      $call = $utt2call{$utt};
      if(exists $call2spk{$call}){
	  $spk = $call2spk{$call};
      }else
      {
	  die "spk for utt ${utt} not found";
      }
      if (exists $call2lang{$call}){
	  $lang = $call2lang{$call};
      }else{
	  die "lang for utt ${utt} not found";
      }
      $futt = "${spk}-${utt}";
      $utt2fixedutt{$utt} = $futt;
      print SPKR "${futt} $spk\n";
      print MODEL "${futt} $model\n";
      print U2L "${futt} $lang\n";
  }
}
close(SPKR) || die;
close(MODEL) || die;
close(U2L) || die;

if (system("find $db_base/data/enrollment/ -name '*.sph' > $tmp_dir_enroll/sph.list") != 0) {
  die "Error getting list of sph files";
}

open(WAVLIST, "<$tmp_dir_enroll/sph.list") or die "cannot open wav list";

while(<WAVLIST>) {
  chomp;
  $sph = $_;
  @t = split("/",$sph);
  @t1 = split("[./]",$t[$#t]);
  $utt=$utt2fixedutt{$t1[0]};
  if($fs == 8) {
      print WAV "$utt"," sph2pipe -f wav -p -c 1 $sph |\n";
  } else {
      print WAV "$utt"," sph2pipe -f wav -p -c 1 $sph | sox -t wav - -t wav -r 16k - |\n";
  }
}
close(WAV) || die;


# Handle test
$out_dir_test= "$out_dir/sre16_eval_test";
if (system("mkdir -p $out_dir_test")) {
  die "Error making directory $out_dir_test";
}

$tmp_dir_test = "$out_dir_test/tmp";
if (system("mkdir -p $tmp_dir_test") != 0) {
  die "Error making directory $tmp_dir_test";
}

open(SPKR, ">$out_dir_test/utt2spk") || die "Could not open the output file $out_dir_test/utt2spk";
open(U2L, ">$out_dir_test/utt2lang") || die "Could not open the output file $out_dir_test/utt2lang";
open(WAV, ">$out_dir_test/wav.scp") || die "Could not open the output file $out_dir_test/wav.scp";
open(TRIALS, ">$out_dir_test/trials") || die "Could not open the output file $out_dir_test/trials";
open(TGL_TRIALS, ">$out_dir_test/trials_tgl") || die "Could not open the output file $out_dir_test/trials_tgl";
open(YUE_TRIALS, ">$out_dir_test/trials_yue") || die "Could not open the output file $out_dir_test/trials_yue";

if (system("find $db_base/data/test/ -name '*.sph' > $tmp_dir_test/sph.list") != 0) {
  die "Error getting list of sph files";
}

open(KEY, "<$db_base/docs/sre16_eval_trial_key.tsv") || die "Could not open trials file $db_base/docs/sre16_eval_trial_key.tsv.  It might be located somewhere else in your distribution.";

open(WAVLIST, "<$tmp_dir_test/sph.list") or die "cannot open wav list";


%utt2futt = ();
while(<WAVLIST>) {
  chomp;
  $sph = $_;
  @t = split("/",$sph);
  @t1 = split("[./]",$t[$#t]);
  $utt=$t1[0];
  $call = $utt2call{$utt};
  $spk=$call2spk{$call};
  $lang=$call2lang{$call};
  $futt="${spk}-${utt}";
  $utt2futt{$utt} = $futt;
  if($fs == 8){
      print WAV "$futt"," sph2pipe -f wav -p -c 1 $sph |\n";
  } else {
      print WAV "$futt"," sph2pipe -f wav -p -c 1 $sph | sox -t wav - -t wav -r 16k - |\n";
  }
  print SPKR "$futt $spk\n";
  print U2L "$futt $lang\n";
}
close(WAV) || die;
close(SPKR) || die;
close(U2L) || die;

while (<KEY>) {
  $line = $_;
  @toks = split(" ", $line);
  $model = $toks[0];
  $utt = $toks[1];
  $call = $utt2call{$utt};
  $futt = $utt2futt{$utt};
  $target_type = $toks[3];
  if ($utt ne "segment") {
    print TRIALS "${model} ${futt} ${target_type}\n";
    if ($call2lang{$call} eq "tgl") {
      print TGL_TRIALS "${model} ${futt} ${target_type}\n";
    } elsif ($call2lang{$call} eq "yue") {
      print YUE_TRIALS "${model} ${futt} ${target_type}\n";
    } else {
      die "Unexpected language $call2lang{$call} for utterance $utt.";
    }
  }
}

close(TRIALS) || die;
close(TGL_TRIALS) || die;
close(YUE_TRIALS) || die;

if (system("utils/utt2spk_to_spk2utt.pl $out_dir_enroll/utt2spk >$out_dir_enroll/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir_enroll";
}
if (system("utils/utt2spk_to_spk2utt.pl $out_dir_enroll/utt2model >$out_dir_enroll/spk2model") != 0) {
  die "Error creating spk2model file in directory $out_dir_enroll";
}
if (system("utils/utt2spk_to_spk2utt.pl $out_dir_test/utt2spk >$out_dir_test/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir_test";
}
if (system("utils/fix_data_dir.sh $out_dir_enroll") != 0) {
  die "Error fixing data dir $out_dir_enroll";
}
if (system("utils/fix_data_dir.sh $out_dir_test") != 0) {
  die "Error fixing data dir $out_dir_test";
}
