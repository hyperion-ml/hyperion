BEGIN{
  while (getline <fscores){
    nf=split($1,f,"@")
    scores[f[1]" "f[2]" "$2]=f[1]"\t"f[2]"\t"$2"\t"$3;
  }
  getline;
  print "modelid\timageid\tsegmentid\tLLR"
}
{ tag=$1" "$2" "$3;
  if (tag in scores) {
    print scores[tag];
  }
  else {
    print "MISSING "tag;
  }
}
