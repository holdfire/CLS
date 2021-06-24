# Support Chinese
apt-get install -y language-pack-zh-hans
export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
localedef -i en_US -f UTF-8 en_US.UTF-8

# git configuration
git config --global user.email "liuxing.a@mininglamp.com"
git config --global user.name "liuxing.a" 