
#!/bin/bash

cat ../dev_files_copy.txt | while read line
do
  echo $line
  cp /projekte/emmy-noether-roth/mist/mt/bow_exps/wikiHow_articles/$line dev-files/.
done
