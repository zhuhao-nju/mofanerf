due 

# check if samples.tar.gz exists
if [ -f ./pretrained_models.tar.gz ] ; then
echo "pretrained_models.tar.gz has already been downloaded."
fi

# download samples.tar.gz
if [ ! -f ./pretrained_models.tar.gz ] ; then
wget --no-check-certificate 'https://drive.google.com/u/0/uc?id=1jlhtm8BoChczSMks3WjChQ_SGLXyZ9ka&export=download' -O ./pretrained_models.tar.gz
fi
# wget --no-check-certificate 'https://box.nju.edu.cn/f/aad38f30d45d41a1a78f/?dl=1' -O ./pretrained_models.tar.gz

# extract files
mkdir -p ./logs/
tar -zxf pretrained_models.tar.gz -C ./logs/
echo "pretrained models have been extracted to ./logs/"

