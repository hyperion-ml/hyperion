#!/usr/bin/perl
# Note: this is an old version script in the commit id 1eb12f2ed01801a50c3f7ba014809bf7c7212f28. NOT process LANG, GENDER, NAT
# Copyright 2018  Ewald Enzinger
#           2018  David Snyder
#           2020  Jesus Villalba
#
# Usage: make_voxceleb1.pl /export/voxceleb1 data/
# Create trial lists for Voxceleb1 original, Entire (E) and hard (H), 
# with cleaned and non-cleaned versions

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

open(META_IN, "<", "$data_base/vox1_meta.csv") or die "Could not open the meta data file $data_base/vox1_meta.csv";
my %id2spkr = ();
while (<META_IN>) {
  chomp;
  my ($vox_id, $spkr_id, $gender, $nation, $set) = split;
  $id2spkr{$vox_id} = $spkr_id;

}
close(META_IN) or die;

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

open(SPKR_TEST, ">", "$out_dir/utt2spk") or die "Could not open the output file $out_dir/utt2spk";
open(WAV_TEST, ">", "$out_dir/wav.scp") or die "Could not open the output file $out_dir/wav.scp";

foreach (@spkr_dirs) {
  my $spkr_id = $_;
  my $new_spkr_id = $spkr_id;
  # If we're using a newer version of VoxCeleb1, we need to "deanonymize"
  # the speaker labels.
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
    my $utt_id = "$new_spkr_id-$rec_id-$segment";
    print WAV_TEST "$utt_id", " $wav", "\n";
    print SPKR_TEST "$utt_id", " $new_spkr_id", "\n";
  }
}

close(SPKR_TEST) or die;
close(WAV_TEST) or die;

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

