# revise apt-get source file
cd /etc/apt/
mv sources.list sources.list.backup
cd -
cp ./run/sources.list ./

# necessary packages
apt-get update
apt-get install -y python3-pip vim git openssh-server lrzsz zip
apt-get install -y libsm6 libxrender1 libxext6 libglib2.0-dev

# Support Chinese
apt-get install -y language-pack-zh-hans
export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
localedef -i en_US -f UTF-8 en_US.UTF-8

# git configuration
git config --global user.email "1028377876@qq.com"
git config --global user.name "liuxing" 

# start ssh service
cd
mkdir .ssh
cd .ssh
cp ./authorized_keys ./
service ssh start 

# python packages
pip3 install -r ./equirements.txt
