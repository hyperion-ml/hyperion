BEGIN{
  while (getline <fscores){
    scores[$1" "$2]=$1"\t"$2"\t"$3;
  }
  getline;
  print "imageid\tsegmentid\tLLR"
}
{ tag=$1" "$2;
  if (tag in scores) {
    print scores[tag];
  }
  else {
    print "MISSING "tag;
  }
}
