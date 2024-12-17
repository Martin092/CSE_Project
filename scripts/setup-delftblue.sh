echo "Script is not ready, dont use on delftblue"

#module load 2023r1
#module load intel/oneapi_2024.0
#module load mpi/2021.11
#
## install newer python version because the version on delftblue was already old during the stone age
#module load curl
#curl https://pyenv.run | bash
#echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
#echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
#echo 'eval "$(pyenv init -)"' >> ~/.bashrc
#module load libffi/3.4.2
#pyenv install 3.11.2
#pyenv global 3.11.2
#
## load pip and create environment
#module load py-pip/22.2.2
#python3.11 -m venv --system-site-packages venv
#source venv/bin/activate
#pip install -r requireme  nts.txt