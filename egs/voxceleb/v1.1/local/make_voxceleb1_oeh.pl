#!/usr/bin/perl
#
# Copyright 2018  Ewald Enzinger
#           2018  David Snyder
#           2020  Jesus Villalba
#
# Usage: make_voxceleb1.pl /export/voxceleb1 data/
# Create trial lists for Voxceleb1 original, Entire (E) and hard (H), 
# with cleaned and non-cleaned versions
# Attention:
#  - This script is for the old version of the dataset without anonymized speaker-ids
#  - This script assumes that the voxceleb1 dataset has all speaker directories
#  dumped in the same wav directory, NOT separated dev and test directories

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-voxceleb1> <path-to-data-dir>\n";
  print STDERR "e.g. $0 /export/voxceleb1 data/\n";
  exit(1);
}

($data_base, $out_dir) = @ARGV;
my $out_dir = "$out_dir/voxceleb1_test";

if (system("mkdir -p $out_dir") != 0) {
  die "Error making directory $out_dir";
}

my $url_base="http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta";
my @trials_basename = ("very_test.txt", "very_test2.txt", "list_test_hard.txt", "list_test_hard2.txt", "list_test_all.txt", "list_test_all2.txt");
my @trials_url = ("$url_base/veri_test.txt", "$url_base/veri_test2.txt", "$url_base/list_test_hard.txt", "$url_base/list_test_hard2.txt", "$url_base/list_test_all.txt", "$url_base/list_test_all2.txt");
my @trials = ("trials_o", "trials_o_clean", "trials_h", "trials_h_clean", "trials_e", "trials_e_clean");

my $meta_url = "https://www.openslr.org/resources/49/vox1_meta.csv";
my $meta_path = "$data_base/vox1_meta.csv";
if (! -e "$meta_path") {
    $meta_path = "$out_dir/vox1_meta.csv";
    system("wget -O $meta_path $meta_url");
}

open(META_IN, "<", "$meta_path") or die "Could not open the meta data file $meta_path";
my %id2spkr = ();
my %spkr2gender = ();
my %spkr2nation = ();
while (<META_IN>) {
    chomp;
    my ($vox_id, $spkr_id, $gender, $nation, $set) = split "\t";
    $id2spkr{$vox_id} = $spkr_id;
    $spkr2gender{$spkr_id} = $gender;
    $nation =~ s@ @-@g;
    $spkr2nation{$spkr_id} = $nation;
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
  my $utt_id = "$spkr_id-$vid_id-00$file_id";
  $utt_id =~ s@\.wav$@@;
  $utt2lang{$utt_id} = $lang;
}
close(LID_IN) or die;

#download trials from voxceleb web page
for($i = 0; $i <= $#trials; $i++) {

    my $file_i = "$out_dir/$trials_basename[$i]";
    my $url_i = $trials_url[$i];
    my $trial_i = "$out_dir/$trials[$i]";
    if (! -e $file_i) {
	system("wget -O $file_i $url_i");
    }
    #mapping from new speaker ids and file-names to old ones
    open(TRIAL_IN, "<", "$file_i") or die "Could not open the verification trials file $file_i";
    open(TRIAL_OUT, ">", "$trial_i") or die "Could not open the output file $trial_i";
    while (<TRIAL_IN>) {
	chomp;
	my ($tar_or_non, $path1, $path2) = split;

	# Create entry for left-hand side of trial
	my ($vox_id, $rec_id, $segment) = split('/', $path1);
	$segment =~ s/\.wav$//;
	my $spkr_id = $id2spkr{$vox_id};
	my $utt_id1 = "$spkr_id-$rec_id-00$segment";
	
	# Create entry for right-hand side of trial
	my ($vox_id, $rec_id, $segment) = split('/', $path2);
	$segment =~ s/\.wav$//;
	my $spkr_id = $id2spkr{$vox_id};
	my $utt_id2 = "$spkr_id-$rec_id-00$segment";
	
	my $target = "nontarget";
	if ($tar_or_non eq "1") {
	    $target = "target";
	}
	print TRIAL_OUT "$utt_id1 $utt_id2 $target\n";
    }
    
    close(TRIAL_IN) or die;
    close(TRIAL_OUT) or die;
    
}


opendir my $dh, "$data_base/voxceleb1_wav" or die "Cannot open directory: $!";
my @spkr_dirs = grep {-d "$data_base/voxceleb1_wav/$_" && ! /^\.{1,2}$/} readdir($dh);
closedir $dh;

open(SPKR, ">", "$out_dir/utt2spk") or die "Could not open the output file $out_dir/utt2spk";
open(WAV, ">", "$out_dir/wav.scp") or die "Could not open the output file $out_dir/wav.scp";
open(GENDER, ">", "$out_dir/spk2gender") or die "Could not open the output file $out_dir/spk2gender";
open(NAT, ">", "$out_dir/spk2nation") or die "Could not open the output file $out_dir/spk2nation";
open(LANG, ">", "$out_dir/utt2lang") or die "Could not open the output file $out_dir/utt2lang";

foreach (@spkr_dirs) {
  my $spkr_id = $_;
  my $new_spkr_id = $spkr_id;
  # If we're using a newer version of VoxCeleb1, we need to "deanonymize"
  # the speaker labels.
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
    my $utt_id = "$new_spkr_id-$rec_id-$segment";
    print WAV "$utt_id", " $wav", "\n";
    print SPKR "$utt_id", " $new_spkr_id", "\n";
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
close(NAT) or die;

if (system(
  "cat $out_dir/trials_* | sort -u > $out_dir/trials") != 0) {
  die "Error creating trials file in directory $out_dir";
}

if (system(
  "awk '{ print \$1,\$1 }' $out_dir/trials | sort -u > $out_dir/utt2model") != 0) {
  die "Error creating utt2model file in directory $out_dir";
}

if (system(
  "utils/utt2spk_to_spk2utt.pl $out_dir/utt2spk >$out_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir";
}
system("env LC_COLLATE=C utils/fix_data_dir.sh $out_dir");
if (system("env LC_COLLATE=C utils/validate_data_dir.sh --no-text --no-feats $out_dir") != 0) {
  die "Error validating directory $out_dir";
}

