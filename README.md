# auto_mine

this program is to auto start a zec miner  
and this program propose a concept  
    1. create a thread to monitor GPU usge (here, monitor GPU MEM)  
    2. if the GPU is available, start miner  
    3. if not, stop the miner  


reminder:  
    to run this program, some python package is required  
    1. pynvml - pip install nvidia-ml-py  
    2. psutil - pip install psutil  

notice**  
there might be a wrong syntax in pynvml, use python3 instead  
