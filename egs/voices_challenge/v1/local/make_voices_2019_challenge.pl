#!/usr/bin/perl
#
# Copyright 2019 David Snyder
#
# Usage: make_voices_2019_challenge.pl /export/corpora/SRI/voices_2019_challenge data

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-voices_2019_challenge> <path-to-data-dir>\n";
  print STDERR "e.g. $0 /export/voices_2019_challenge dev data/dev\n";
  exit(1);
}

($data_base, $out_dir) = @ARGV;

opendir my $dh, "$data_base/sid_dev" or die "Cannot open directory: $!";
my @spkr_dirs = grep {-d "$data_base/sid_dev/$_" && ! /^\.{1,2}$/} readdir($dh);
closedir $dh;

if (system("mkdir -p $out_dir/dev_enroll") != 0) {
  die "Error making directory $out_dir/dev_enroll";
}

if (system("mkdir -p $out_dir/dev_test") != 0) {
  die "Error making directory $out_dir/dev_test";
}

my %enroll_utts = ();
my %test_utts = ();

open(ENROLL, "<", "$data_base/sid_dev_lists_and_keys/dev-enroll.lst") or die "Could not open $data_base/sid_dev_lists_and_keys/dev-enroll.lst";
while (<ENROLL>) {
  chomp;
  @toks = split(" ", $_);
  $utt = $toks[0];
  $enroll_utts{$utt} = ();
}
close(ENROLL) or die;

open(TEST, "<", "$data_base/sid_dev_lists_and_keys/dev-test.lst") or die "Could not open $data_base/sid_dev_lists_and_keys/dev-test.lst";
while (<TEST>) {
  chomp;
  $utt = $_;
  $test_utts{$utt} = ();
}
close(TEST) or die;

open(TRIALS_IN, "<", "$data_base/sid_dev_lists_and_keys/dev-trial-keys.lst") or die "Could not open $data_base/sid_dev_lists_and_keys/dev-trial-keys.lst";
open(TRIALS_OUT, ">", "$out_dir/dev_test/trials") or die "Could not open the output file $out_dir/dev_test/trials";
while (<TRIALS_IN>) {
  chomp;
  @toks = split(" ", $_);
  $enroll = $toks[0];
  $test = $toks[1];
  $type = $toks[2];
  if ($type eq "tgt") {
    print TRIALS_OUT "$enroll $test target\n";
  } else {
    print TRIALS_OUT "$enroll $test nontarget\n";
  }
}
close(TRIALS_IN) or die;
close(TRIALS_OUT) or die;

open(SPKR_ENROLL, ">", "$out_dir/dev_enroll/utt2spk") or die "Could not open the output file $out_dir/dev_enroll/utt2spk";
open(WAV_ENROLL, ">", "$out_dir/dev_enroll/wav.scp") or die "Could not open the output file $out_dir/dev_enroll/wav.scp";
open(SPKR_TEST, ">", "$out_dir/dev_test/utt2spk") or die "Could not open the output file $out_dir/dev_test/utt2spk";
open(WAV_TEST, ">", "$out_dir/dev_test/wav.scp") or die "Could not open the output file $out_dir/dev_test/wav.scp";

foreach (@spkr_dirs) {
  my $spkr_id = $_;

  opendir my $dh, "$data_base/sid_dev/$spkr_id/" or die "Cannot open directory: $!";
  my @files = map{s/\.[^.]+$//;$_}grep {/\.wav$/} readdir($dh);
  closedir $dh;

  foreach (@files) {
    my $name = $_;
    my $wav = "$name.wav";
    my $utt_id = "$name";
    my $test_utt_id = "sid_dev/$spkr_id/$name.wav";
    if (exists($enroll_utts{$utt_id})) {
      print WAV_ENROLL "$utt_id", " $data_base/sid_dev/$spkr_id/$wav", "\n";
      print SPKR_ENROLL "$utt_id", " $spkr_id", "\n";
    }
    if (exists($test_utts{$test_utt_id})) {
      print WAV_TEST "$test_utt_id", " $data_base/sid_dev/$spkr_id/$wav", "\n";
      print SPKR_TEST "$test_utt_id", " $test_utt_id", "\n";
    }
  }
}
close(SPKR_ENROLL) or die;
close(WAV_ENROLL) or die;
close(SPKR_TEST) or die;
close(WAV_TEST) or die;

if (system(
  "utils/utt2spk_to_spk2utt.pl $out_dir/dev_enroll/utt2spk >$out_dir/dev_enroll/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir/dev_enroll";
}

system("env LC_COLLATE=C utils/fix_data_dir.sh $out_dir/dev_enroll");
#if (system("env LC_COLLATE=C utils/validate_data_dir.sh --no-text --no-feats $out_dir/dev_enroll") != 0) {
#  die "Error validating directory $out_dir/dev_enroll";
#}

if (system(
  "utils/utt2spk_to_spk2utt.pl $out_dir/dev_test/utt2spk >$out_dir/dev_test/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir/dev_test";
}

system("env LC_COLLATE=C utils/fix_data_dir.sh $out_dir/dev_test");
#if (system("env LC_COLLATE=C utils/validate_data_dir.sh --no-text --no-feats $out_dir/dev_test") != 0) {
#  die "Error validating directory $out_dir/dev_test";
#}
