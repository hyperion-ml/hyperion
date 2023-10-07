#!/usr/bin/perl
#
# Copyright 2018  Ewald Enzinger
#           2018  David Snyder
#           2018  Jesus Villalba
#
# Apache 2.0
# Usage: make_voxceleb1cat.pl /export/voxceleb1 data/
# Attention:
#  - This script is for the old version of the dataset without anonymized speaker-ids
#  - This version of the script does NOT remove SITW overlap speakers
#  - Files from the same video are concatenated into 1 segment
#  - This script assumes that the voxceleb1 dataset has all speaker directories
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
  $id2spkr{$vox_id} = $spkr_id;
  $spkr2gender{$spkr_id} = $gender;
  $nation =~ s@ @-@g;
  $spkr2nation{$spkr_id} = $nation;
  if ( $set eq "test"){
      $test_spkrs{$spkr_id} = ();
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
  my ($vox_id, $vid_id, $file_id) = split '/', $utt_id;
  my $spkr_id = $id2spkr{$vox_id};
  my $utt_id = "$spkr_id-$vid_id";
  $utt2lang{$utt_id} = $lang;
}
close(LID_IN) or die;


opendir my $dh, "$data_base/voxceleb1_wav" or die "Cannot open directory: $!";
my @spkr_dirs = grep {-d "$data_base/voxceleb1_wav/$_" && ! /^\.{1,2}$/} readdir($dh);
closedir $dh;

open(GENDER, ">", "$out_dir/spk2gender") or die "Could not open the output file $out_dir/spk2gender";
open(NAT, ">", "$out_dir/spk2nation") or die "Could not open the output file $out_dir/spk2nation";

my %rec2utt = ();
my %rec2spk = ();
foreach (@spkr_dirs) {
    my $spkr_id = $_;
    my $new_spkr_id = $spkr_id;
    if (exists $id2spkr{$spkr_id}) {
	$new_spkr_id = $id2spkr{$spkr_id};
    }
    print GENDER "$new_spkr_id $spkr2gender{$new_spkr_id}\n";
    print NAT "$new_spkr_id $spkr2nation{$new_spkr_id}\n";

    opendir my $dh, "$data_base/voxceleb1_wav/$spkr_id/" or die "Cannot open directory: $!";
    my @files = map{s/\.[^.]+$//;$_}grep {/\.wav$/} readdir($dh);
    closedir $dh;
    foreach (@files) {
	my $filename = $_;
	my $rec_id = substr($filename, 0, 11);
	my $segment = substr($filename, 12, 7);
	my $wav = "$data_base/voxceleb1_wav/$spkr_id/$filename.wav";
	my $utt_id = "$new_spkr_id-$rec_id";
	if (not exists $test_spkrs{$new_spkr_id}) {
	    if (not exists $rec2utt{$utt_id}) {
		$rec2spk{$utt_id} = $new_spkr_id;
		$rec2utt{$utt_id} = $wav
	    }
	    else {
		$rec2utt{$utt_id} = $rec2utt{$utt_id} . " " . $wav
	    }
	}
    }
}
close(GENDER) or die;
close(NAT) or die;

open(SPKR, ">", "$out_dir/utt2spk") or die "Could not open the output file $out_dir/utt2spk";
open(WAV, ">", "$out_dir/wav.scp") or die "Could not open the output file $out_dir/wav.scp";
open(LANG, ">", "$out_dir/utt2lang") or die "Could not open the output file $out_dir/utt2lang";

foreach my $utt_id (keys %rec2spk) {
    my $wav = "";
    if($fs == 8){
	$wav = "sox " . $rec2utt{$utt_id} . " -t wav -r 8k - |";
    }
    else{
	$wav = "sox " . $rec2utt{$utt_id} . " -t wav - |";
    }
    my $spkr_id = $rec2spk{$utt_id};
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
