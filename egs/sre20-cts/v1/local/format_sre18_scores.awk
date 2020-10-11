BEGIN{
  while (getline <fscores){
    scores[$1" "$2]=$1"\t"$2"\ta\t"$3+bias;
  }
  getline;
  print "modelid\tsegmentid\tside\tLLR"
}
{ tag=$1" "$2;
  if (tag in scores) {
    print scores[tag];
  }
  else {
    print "MISSING "tag;
  }
}
