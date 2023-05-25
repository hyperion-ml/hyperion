#!/usr/bin/perl
#
# Copyright 2018  Johns Hopkins University (Jesus Villalba)
# Copyright 2018  Ewald Enzinger
#
# Apache 2.0
# Usage: make_voxceleb2cat.pl /export/voxceleb2cat_train dev 16 data/dev
#
# Note: This script requires ffmpeg to be installed and its location included in $PATH.

if (@ARGV != 4) {
  print STDERR "Usage: $0 <path-to-voxceleb2> <dataset> fs <path-to-data-dir>\n";
  print STDERR "e.g. $0 /export/voxceleb2 dev 16 data/dev\n";
  exit(1);
}

# Check that ffmpeg is installed.
if (`which ffmpeg` eq "") {
  die "Error: this script requires that ffmpeg is installed.";
}

($data_base, $dataset, $fs, $out_dir) = @ARGV;

print "Preparing VoxCeleb2 Cat in $out_dir \n";

if ("$dataset" ne "dev" && "$dataset" ne "test") {
  die "dataset parameter must be 'dev' or 'test'!";
}

my $dataset_path = "" ;
if ( -d "$data_base/$dataset/aac" ){
    $dataset_path = "$data_base/$dataset/aac"
}
else {
    $dataset_path = "$data_base/$dataset"
}


if (system("mkdir -p $out_dir") != 0) {
  die "Error making directory $out_dir";
}

if (system("mkdir -p $out_dir/lists_cat") != 0) {
  die "Error making directory $out_dir/lists_cat";
}

print "Reading metadata\n";
my $meta_url = "https://www.openslr.org/resources/49/vox2_meta.csv";
my $meta_path = "$data_base/vox2_meta.csv";
if (! -e "$meta_path") {
    $meta_path = "$out_dir/vox2_meta.csv";
    system("wget --no-check-certificate -O $meta_path $meta_url");
}
open(META_IN, "<", "$meta_path") or die "Could not open the output file $meta_path";
my %spkr2gender = ();
while (<META_IN>) {
  chomp;
  my ($spkr, $vox_id, $vgg_id, $gender, $set) = split;
  $spkr2gender{$vox_id} = $gender;
}
close(META_IN) or die;

print "Reading languages estimated voxlingua \n";
my $lid_url = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2021/lang_vox2_final.csv";
my $lid_path = "$data_base/lang_vox2_final.csv";
if (! -e "$lid_path") {
    $lid_path = "$out_dir/lang_vox2_final.csv";
    system("wget -O $lid_path $lid_url");
}
open(LID_IN, "<", "$lid_path") or die "Could not open the output file $lid_path";
my %utt2lang = ();
while (<LID_IN>) {
  chomp;
  my ($utt_id, $lang, $score) = split ',';
  $utt_id =~ s@/@-@g;
  $utt_id =~ s@-[^-]*\.wav$@@;
  $utt2lang{$utt_id} = $lang;
}
close(LID_IN) or die;

open(SPKR, ">", "$out_dir/utt2spk") or die "Could not open the output file $out_dir/utt2spk";
open(WAV, ">", "$out_dir/wav.scp") or die "Could not open the output file $out_dir/wav.scp";
open(LANG, ">", "$out_dir/utt2lang") or die "Could not open the output file $out_dir/utt2lang";
open(GENDER, ">", "$out_dir/spk2gender") or die "Could not open the output file $out_dir/spk2gender";

opendir my $dh, "$dataset_path" or die "Cannot open directory: $!";
my @spkr_dirs = grep {-d "$dataset_path/$_" && ! /^\.{1,2}$/} readdir($dh);
closedir $dh;

my $num_spkrs = @spkr_dirs;
my $count = 0;
foreach (@spkr_dirs) {
  my $spkr_id = $_;

  $count++ ;
  print "  processing speaker $spkr_id $count / $num_spkrs \n";
  print GENDER "$spkr_id $spkr2gender{$spkr_id}\n";

  opendir my $dh, "$dataset_path/$spkr_id/" or die "Cannot open directory: $!";
  my @rec_dirs = grep {-d "$dataset_path/$spkr_id/$_" && ! /^\.{1,2}$/} readdir($dh);
  closedir $dh;

  foreach (@rec_dirs) {
      my $rec_id = $_;
      my $utt_id = "$spkr_id-$rec_id";
      my $file_list = "$out_dir/lists_cat/$utt_id.txt";
      if (system("find $dataset_path/$spkr_id/$rec_id -name \"*.m4a\" -printf \"file %p\\n\" > $file_list") != 0){
	  die "Error creating $file_list";
      }
      my $wav = "ffmpeg -v 8 -f concat -safe 0 -i $file_list -f wav -acodec pcm_s16le -|";
      if($fs == 8){
	  $wav = $wav." sox -t wav - -t wav -r 8k - |"
      }
      print WAV "$utt_id", " $wav", "\n";
      print SPKR "$utt_id", " $spkr_id", "\n";
      if (exists $utt2lang{$utt_id}) {
	  print LANG "$utt_id", " $utt2lang{$utt_id}", "\n";
      }
      else {
	  print LANG "$utt_id N/A\n";
      }
  }
}
close(SPKR) or die;
close(WAV) or die;
close(LANG) or die;
close(GENDER) or die;

if (system(
  "utils/utt2spk_to_spk2utt.pl $out_dir/utt2spk >$out_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir";
}
system("env LC_COLLATE=C utils/fix_data_dir.sh $out_dir");
if (system("env LC_COLLATE=C utils/validate_data_dir.sh --no-text --no-feats $out_dir") != 0) {
  die "Error validating directory $out_dir";
}
