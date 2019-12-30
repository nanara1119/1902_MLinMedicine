# ECG Classification 
Implementation<br> https://www.nature.com/articles/s41591-018-0268-3<br>
Machine Learning in Medicine lecture<br>(https://sites.google.com/view/bmilabsaihstskku/home/lectures/2019-ml-for-medicine?authuser=0)<br> 
Reference at https://github.com/awni/ecg<br>

#
# Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network

<H2>Windows Setting</H2> 
1. install Python 3.7<br> 
2. pip install --user virtualenv<br>
3. cd Project<br>
4. python -m venv ./venv<br>
5. .\venv\Scripts\activate<br>
6. pip install -r requirements.txt<br>

<H2>Data Download</H2>
1. python data.py

<H2>Train</H2>
1. python train.py (optional) --epochs 1 --batchsize 32<br>
2. data\model\ (check result folder)

<H2>Predict</H2>
1. python predict.py --model [saved model path]<br>
    ex) python predict.py --model data\model\base.hdf5








