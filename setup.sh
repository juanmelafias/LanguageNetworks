#Install virtualenv package
python -m pip install virtualenv

#Create virtual environment
python -m virtualenv venv

#Activate virtual env
source venv/Scripts/activate 

#Install requirement packages
pip install --upgrade pip
pip install -r reqs/requirements.txt
pip install -r reqs/requirements-test.txt
pip install -r reqs/requirements-nb.txt