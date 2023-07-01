#!/usr/bin/perl
#
# Copyright 2018  Ewald Enzinger
#           2018  David Snyder
#           2018  Jesus Villalba
#
# Apache 2.0
# Usage: make_voxceleb1cat_v2.pl /export/voxceleb1 data/
# Attention:
# - This script is for the recent version of the dataset
# - This version of the script does NOT remove SITW overlap speakers
# - Files from the same video are concatenated into 1 segment
# - This script assumes that the voxceleb1 dataset has all speaker directories
#  dumped in the same wav directory, NOT separated dev and test directories

if (@ARGV != 3) {
  print STDERR "Usage: $0 <path-to-voxceleb1> fs <path-to-data-dir>\n";
  print STDERR "e.g. $0 /export/voxceleb1 16 data/\n";
  exit(1);
}

($data_base, $fs, $out_dir) = @ARGV;
my $out_dir = "$out_dir/voxceleb1cat_train";

if (system("mkdir -p $out_dir") != 0) {
  die "Error making directory $out_train_dir";
}

my $meta_url = "https://www.openslr.org/resources/49/vox1_meta.csv";
my $meta_path = "$data_base/vox1_meta.csv";
if (! -e "$meta_path") {
    $meta_path = "$out_dir/vox1_meta.csv";
    system("wget -O $meta_path $meta_url");
}

open(META_IN, "<", "$meta_path") or die "Could not open the meta data file $meta_path";

my %id2spkr = ();
my $test_spkrs = ();
my %spkr2gender = ();
my %spkr2nation = ();
while (<META_IN>) {
  chomp;
  my ($vox_id, $spkr_id, $gender, $nation, $set) = split "\t";
  $spkr2gender{$vox_id} = $gender;
  $nation =~ s@ @-@g;
  $spkr2nation{$vox_id} = $nation;
  if ( $set eq "test"){
      $test_spkrs{$vox_id} = ();
  }
}
close(META_IN) or die;

my $lid_url = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2021/lang_vox1_final.csv";
my $lid_path = "$data_base/lang_vox1_final.csv";
if (! -e "$lid_path") {
    $lid_path = "$out_dir/lang_vox1_final.csv";
    system("wget -O $lid_path $lid_url");
}
open(LID_IN, "<", "$lid_path") or die "Could not open the output file $lid_path";
my %utt2lang = ();
while (<LID_IN>) {
  chomp;
  my ($utt_id, $lang, $score) = split ',';
  my ($spkr_id, $vid_id, $file_id) = split '/', $utt_id;
  my $utt_id = "$spkr_id-$vid_id";
  $utt2lang{$utt_id} = $lang;
}
close(LID_IN) or die;

my $wav_dir = "$data_base/wav";
opendir my $dh, "$wav_dir" or die "Cannot open directory: $!";
my @spkr_dirs = grep {-d "$wav_dir/$_" && ! /^\.{1,2}$/ || -l "$wav_dir/$_" } readdir($dh);
closedir $dh;

open(GENDER, ">", "$out_dir/spk2gender") or die "Could not open the output file $out_dir/spk2gender";
open(NAT, ">", "$out_dir/spk2nation") or die "Could not open the output file $out_dir/spk2nation";

my %utt2wav = ();
my %utt2spk = ();
foreach (@spkr_dirs) {
    my $spkr_id = $_;

    print GENDER "$spkr_id $spkr2gender{$spkr_id}\n";
    print NAT "$spkr_id $spkr2nation{$spkr_id}\n";

    my $spkr_dir = "$wav_dir/$spkr_id";
    opendir my $dh, "$spkr_dir" or die "Cannot open directory: $!";
    my @vid_dirs = grep {-d "$spkr_dir/$_" && ! /^\.{1,2}$/ } readdir($dh);
    my @files = map{s/\.[^.]+$//;$_}grep {/\.wav$/} readdir($dh);
    closedir $dh;
    foreach (@vid_dirs) {
	my $vid_id = $_;
	my $vid_dir = "$spkr_dir/$vid_id";
	opendir my $dh, "$vid_dir" or die "Cannot open directory: $!";
	my @files = map{s/\.[^.]+$//;$_}grep {/\.wav$/} readdir($dh);
	closedir $dh;
	foreach (@files) {
	    my $segment = $_;
	    my $wav = "$vid_dir/$segment.wav";
	    my $utt_id = "$spkr_id-$vid_id";
	    if (not exists $test_spkrs{$spkr_id}) {
		if (not exists $utt2wav{$utt_id}) {
		    $utt2spk{$utt_id} = $spkr_id;
		    $utt2wav{$utt_id} = $wav
		}
		else {
		    $utt2wav{$utt_id} = $utt2wav{$utt_id} . " " . $wav
		}
	    }
	}
    }
}
close(GENDER) or die;
close(NAT) or die;

open(SPKR, ">", "$out_dir/utt2spk") or die "Could not open the output file $out_dir/utt2spk";
open(WAV, ">", "$out_dir/wav.scp") or die "Could not open the output file $out_dir/wav.scp";
open(LANG, ">", "$out_dir/utt2lang") or die "Could not open the output file $out_dir/utt2lang";

foreach my $utt_id (keys %utt2spk) {
    my $wav = "";
    if($fs == 8){
	$wav = "sox " . $utt2wav{$utt_id} . " -t wav -r 8k - |";
    }
    else{
	$wav = "sox " . $utt2wav{$utt_id} . " -t wav - |";
    }
    my $spkr_id = $utt2spk{$utt_id};
    my $land_id = $utt2lang{$utt_id};
    print WAV "$utt_id", " $wav", "\n";
    print SPKR "$utt_id", " $spkr_id", "\n";
    if (exists $utt2lang{$utt_id}) {
	print LANG "$utt_id", " $utt2lang{$utt_id}", "\n";
    }
    else {
	print LANG "$utt_id N/A\n";
    }
}

close(SPKR) or die;
close(WAV) or die;
close(LANG) or die;

if (system(
  "utils/utt2spk_to_spk2utt.pl $out_dir/utt2spk >$out_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir";
}
system("env LC_COLLATE=C utils/fix_data_dir.sh $out_dir");
if (system("env LC_COLLATE=C utils/validate_data_dir.sh --no-text --no-feats $out_dir") != 0) {
  die "Error validating directory $out_dir";
}

if (system(
  "utils/utt2spk_to_spk2utt.pl $out_dir/utt2spk >$out_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir";
}

system("env LC_COLLATE=C utils/fix_data_dir.sh $out_dir");
if (system("env LC_COLLATE=C utils/validate_data_dir.sh --no-text --no-feats $out_dir") != 0) {
  die "Error validating directory $out_dir";
}
