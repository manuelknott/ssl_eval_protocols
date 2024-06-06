#!/bin/bash

# Download Imagenet

parse_yaml() {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\)\($w\)$s:$s\"\(.*\)\"$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}

eval $(parse_yaml paths.yaml "paths_")

mkdir -p "${paths_datasets_imagenet_d}"


if [ "$1" = "clipart" ]; then
cd "${paths_datasets_imagenet_d}" &&
wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip && unzip clipart.zip && rm clipart.zip &&
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_train.txt &&
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_test.txt
fi

if [ "$1" = "infograph" ]; then
cd "${paths_datasets_imagenet_d}" &&
wget http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip && unzip infograph.zip && rm infograph.zip &&
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_train.txt &&
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_test.txt
fi

if [ "$1" = "painting" ]; then
cd "${paths_datasets_imagenet_d}" &&
wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip && unzip painting.zip && rm painting.zip &&
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_train.txt &&
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_test.txt
fi

if [ "$1" = "quickdraw" ]; then
cd "${paths_datasets_imagenet_d}" &&
wget http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip && unzip quickdraw.zip && rm quickdraw.zip &&
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_train.txt &&
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_test.txt
fi

if [ "$1" = "real" ]; then
cd "${paths_datasets_imagenet_d}" &&
wget http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip && unzip real.zip && rm real.zip &&
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_train.txt &&
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_test.txt
fi

if [ "$1" = "sketch" ]; then
cd "${paths_datasets_imagenet_d}" &&
wget http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip && unzip sketch.zip && rm sketch.zip &&
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_train.txt &&
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_test.txt
fi