#!/usr/bin/perl
use warnings; #sed replacement for -w perl parameter
# Copyright 2019   Johns Hopkins University (Jesus Villalba)
# Copyright 2017   Johns Hopkins University (David Snyder)
# Apache 2.0
# Prepares Mixer 6 (LDC2013S03) speech from a specified microphone and
# adjusts sampling frequency to 8k or 16k

if (@ARGV != 4) {
  print STDERR "Usage: $0 <path-to-LDC2013S03> <channel> <f_sample> <path-to-output>\n";
  print STDERR "e.g. $0 /export/corpora5/LDC/LDC2013S03 02 16 data/\n";
  exit(1);
}
($db_base, $ch, $fs, $out_dir) = @ARGV;

@bad_channels = ("01", "03", "14");
if (/$ch/i ~~ @bad_channels) {
  print STDERR "Bad channel $ch\n";
  exit(1);
}

if (-d "$db_base/mx6_speech") {
    $db_base="$db_base/mx6_speech"
}

if (! -d "$db_base/data/pcm_flac/CH$ch/") {
  print STDERR "Directory $db_base/data/pcm_flac/CH$ch/ doesn't exist\n";
  exit(1);
}

$out_dir = "$out_dir/mx6_mic_$ch";
if (system("mkdir -p $out_dir")) {
  print STDERR "Error making directory $out_dir\n";
  exit(1);
}

if (system("mkdir -p $out_dir") != 0) {
  print STDERR "Error making directory $out_dir\n";
  exit(1);
}

open(SUBJECTS, "<$db_base/docs/mx6_subjs.csv") || die "cannot open $$db_base/docs/mx6_subjs.csv";
open(SPKR, ">$out_dir/utt2spk") || die "Could not open the output file $out_dir/utt2spk";
open(U2C, ">$out_dir/utt2clean") || die "Could not open the output file $out_dir/utt2clean";
open(U2L, ">$out_dir/utt2lang") || die "Could not open the output file $out_dir/utt2lang";
open(U2I, ">$out_dir/utt2info") || die "Could not open the output file $out_dir/utt2info";
open(GNDR, ">$out_dir/spk2gender") || die "Could not open the output file $out_dir/spk2gender";
open(WAV, ">$out_dir/wav.scp") || die "Could not open the output file $out_dir/wav.scp";
open(META, "<$db_base/docs/mx6_ivcomponents.csv") || die "cannot open $db_base/docs/mx6_ivcomponents.csv";

%genders;
while (<SUBJECTS>) {
  chomp;
  $line = $_;
  @toks = split(",", $line);
  $spk = $toks[0];
  $gender = lc $toks[1];
  $genders{$spk}=$gender;
  if ($gender eq "f" or $gender eq "m") {
    print GNDR "$spk $gender\n";
  }
}

$num_good_files = 0;
$num_bad_files = 0;
while (<META>) {
  chomp;
  $line = $_;
  @toks = split(",", $line);
  $flac = "$db_base/data/pcm_flac/CH$ch/$toks[0]_CH$ch.flac";
  $t1 = $toks[7];
  $t2 = $toks[8];
  @toks2 = split(/_/, $toks[0]);
  $ldc_id=$toks[0];
  $spk = $toks2[3];
  $room = $toks2[2];
  $gender = $genders{$spk};
  $utt = "${spk}-MX6-$toks2[0]-$toks2[1]-$ch";
  $utt_clean = "${spk}-MX6-$toks2[0]-$toks2[1]-02";
  if (-f $flac) {
      print SPKR "${utt} $spk\n";
      print U2C "${utt} ${utt_clean}\n";
      print U2L "${utt} ENG\n";
      if ($fs == 8) {
	  print WAV "${utt} sox -t flac $flac -r 8k -t wav - trim $t1 =$t2 |\n";
      }
      else {
	  print WAV "${utt} sox -t flac $flac -r 16k -t wav - trim $t1 =$t2 |\n";
      }
      print U2I "${utt} a ${spk} ${gender} ${ldc_id} ENG All_ENG mic phonecall ${ch} ${room} no_alteration not_reported\n";
      $num_good_files++;
  } else {
      print STDERR "File $flac doesn't exist\n";
      $num_bad_files++;
  }
}

print STDERR "Processed $num_good_files utterances; $num_bad_files had missing flac data.\n";

close(SUBJECTS) || die;
close(GNDR) || die;
close(SPKR) || die;
close(U2C) || die;
close(U2L) || die;
close(U2I) || die;
close(WAV) || die;
close(META) || die;

if (system(
  "utils/utt2spk_to_spk2utt.pl $out_dir/utt2spk >$out_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir";
}

system("utils/fix_data_dir.sh --utt-extra-files utt2clean $out_dir");
if (system("utils/validate_data_dir.sh --no-text --no-feats $out_dir") != 0) {
  die "Error validating directory $out_dir";
}
