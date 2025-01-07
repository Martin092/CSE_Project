echo "Script is not ready, dont use on delftblue"

module load 2024r1
module load intel/oneapi_2024.0
module load mpi/2021.11
module load python/3.10.12
module load py-ase/3.21.1
module load py-contourpy/1.0.7
module load py-cycler/0.11.0
module load py-fonttools/4.39.4
module load py-joblib/1.2.0
module load py-kiwisolver/1.4.5
module load py-matplotlib/3.7.1
#module load py-numpy/1.26.1
module load py-packaging/23.1
module load py-pillow/10.0.0
module load py-pluggy/1.0.0
module load py-pyparsing/3.0.9
module load py-python-dateutil/2.8.2
#module load py-scikit-learn/1.3.2
#module load py-scipy/1.11.3
module load py-six/1.16.0
module load py-threadpoolctl/3.1.0
module load py-requests/2.31.0
module load openmpi/4.1.6
module load py-mpi4py/3.1.4
module load py-pip/23.1.2
pip install numpy==2.1.3
pip install scipy==1.14.1
pip install scikit-learn==1.5.2


#module load curl
#curl https://pyenv.run | bash
#echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
#echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
#echo 'eval "$(pyenv init -)"' >> ~/.bashrc
#module load libffi/3.4.2
#module load bzip2/1.0.8
#module load ncurses/6.3
#module load readline/8.1.2
#
## install newer python version because the version on delftblue was already old during the stone age
#pyenv install 3.11.2
#pyenv global 3.11.2
#
## load pip and create environment
#module load py-pip/22.2.2
#python3.11 -m venv --system-site-packages venv
#source venv/bin/activate
#pip install -r requirements.txt