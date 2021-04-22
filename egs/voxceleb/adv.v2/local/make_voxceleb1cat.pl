#!/usr/bin/perl
#
# Copyright 2018  Ewald Enzinger
#           2018  David Snyder
#           2018  Jesus Villalba
#
# Apache 2.0
# Usage: make_voxceleb1cat.pl /export/voxceleb1 data/
# This version of the script does NOT remove SITW overlap speakers
# Files from the same video are concatenated into 1 segment

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

opendir my $dh, "$data_base/voxceleb1_wav" or die "Cannot open directory: $!";
my @spkr_dirs = grep {-d "$data_base/voxceleb1_wav/$_" && ! /^\.{1,2}$/} readdir($dh);
closedir $dh;

if (! -e "$data_base/voxceleb1_test.txt") {
  system("wget -O $data_base/voxceleb1_test.txt http://www.openslr.org/resources/49/voxceleb1_test.txt");
}

if (! -e "$data_base/vox1_meta.csv") {
  system("wget -O $data_base/vox1_meta.csv http://www.openslr.org/resources/49/vox1_meta.csv");
}

open(META_IN, "<", "$data_base/vox1_meta.csv") or die "Could not open the meta data file $data_base/vox1_meta.csv";

my %id2spkr = ();
my $test_spkrs = ();
while (<META_IN>) {
  chomp;
  my ($vox_id, $spkr_id, $gender, $nation, $set) = split;
  $id2spkr{$vox_id} = $spkr_id;
  if ( $set == "test"){
      $test_spkrs{$spkr_id} = ();
  }
}


my %rec2utt = ();
my %rec2spk = ();
foreach (@spkr_dirs) {
    my $spkr_id = $_;
    my $new_spkr_id = $spkr_id;
    if (exists $id2spkr{$spkr_id}) {
	$new_spkr_id = $id2spkr{$spkr_id};
    }
    opendir my $dh, "$data_base/voxceleb1_wav/$spkr_id/" or die "Cannot open directory: $!";
    my @files = map{s/\.[^.]+$//;$_}grep {/\.wav$/} readdir($dh);
    closedir $dh;
    foreach (@files) {
	my $filename = $_;
	my $rec_id = substr($filename, 0, 11);
	my $segment = substr($filename, 12, 7);
	my $wav = "$data_base/voxceleb1_wav/$spkr_id/$filename.wav";
	my $utt_id = "$spkr_id-$rec_id";
	if (not exists $test_spkrs{$new_spkr_id}) {
	    if (not exists $rec2utt{$utt_id}) {
		$rec2spk{$utt_id} = $spkr_id;
		$rec2utt{$utt_id} = $wav
	    }
	    else {
		$rec2utt{$utt_id} = $rec2utt{$utt_id} . " " . $wav
	    }
	}
    }
}

open(SPKR, ">", "$out_dir/utt2spk") or die "Could not open the output file $out_dir/utt2spk";
open(WAV, ">", "$out_dir/wav.scp") or die "Could not open the output file $out_dir/wav.scp";

foreach my $utt_id (keys %rec2spk) {
    my $wav = "";
    if($fs == 8){
	$wav = "sox " . $rec2utt{$utt_id} . " -t wav -r 8k - |";
    }
    else{
	$wav = "sox " . $rec2utt{$utt_id} . " -t wav - |";
    }
    my $spkr_id = $rec2spk{$utt_id};
    print WAV "$utt_id", " $wav", "\n";
    print SPKR "$utt_id", " $spkr_id", "\n";
}

close(SPKR) or die;
close(WAV) or die;

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
